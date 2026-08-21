"""
Value of information (EVPPI) and probability of decision change, via the
smoothing-based estimator of Straub et al. (2025), "Sensitivity measures
for engineering and environmental decision support" (arXiv:2507.08488),
Section 6.1.

Implements their Eq. 30 (the estimator they recommend over Eq. 29): the
smoothing estimator S(x_i, d) is used ONLY to determine the conditionally-
optimal decision at each sample; the actual, unsmoothed utility is then
used to evaluate both terms of the EVPPI difference. This avoids folding
smoothing bias/noise into the utility magnitude itself.

The smoothing estimator S is fit separately per input, using a method
appropriate to that input's type:
  - Categorical risk inputs (calibration method, warming level, SSP,
    vuln1, vuln2 - each with only 2-3 discrete levels): the exact
    conditional group mean. No smoothing model is needed here, since
    there's no natural notion of "nearby" categories to borrow strength
    from - the group mean already IS the conditional expectation,
    estimated as precisely as the sample allows.
  - Continuous decision inputs (CD, AC_d, E_d): LOESS, matching the
    smoothing method used in both of Straub et al.'s worked examples.
"""
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

import config as cfg
from sampling import EpistemicSamples
from utility import compute_utilities

# Risk inputs are categorical (few discrete levels) even though vuln1/vuln2
# are numeric-valued strings - same treatment as PAWN's conditioning groups.
CATEGORICAL_INPUTS = {
    "Calibration method",
    "Warming level",
    "SSP",
    "Vulnerability parameter 1",
    "Vulnerability parameter 2",
}


def _smooth_categorical(x, u_d, x_eval):
    """Exact conditional-group-mean smoother for a categorical input."""
    x = np.asarray(x)
    u_d = np.asarray(u_d, dtype=float)
    group_means = {val: u_d[x == val].mean() for val in np.unique(x)}
    return np.array([group_means[val] for val in np.asarray(x_eval)])


def _smooth_continuous(x, u_d, x_eval, frac):
    """LOESS smoother for a continuous input, evaluated at x_eval."""
    x = np.asarray(x, dtype=float)
    u_d = np.asarray(u_d, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    return lowess(u_d, x, frac=frac, xvals=x_eval, is_sorted=False)


def fit_smoothing_estimator(x, u_d, input_label: str, x_eval=None, frac: float = 0.3):
    """S(x_i, d): the smoothing estimator for one input/decision, evaluated at x_eval.

    x_eval defaults to x itself (i.e. evaluate at the sample's own points).
    """
    if x_eval is None:
        x_eval = x
    if input_label in CATEGORICAL_INPUTS:
        return _smooth_categorical(x, u_d, x_eval)
    return _smooth_continuous(x, u_d, x_eval, frac)


@dataclass
class VoIResult:
    """EVPPI and probability-of-decision-change for one epistemic input.

    input_label : str
    evppi : float
        Expected value of partial perfect information, in raw utility
        units (here, GBP - since utility.compute_utilities is
        financial-only, u = -Y_e(d)).
    prob_change : float
        Pr(d_opt|X_i != d_opt) - probability that learning X_i would
        change the decision from the status-quo optimum.
    optimal_decision_uncertain : int
        d_opt, the status-quo optimal decision (same for every input at
        a given location - included here for convenience/plotting).
    """
    input_label: str
    evppi: float
    prob_change: float
    optimal_decision_uncertain: int


def compute_evppi(samples: EpistemicSamples, input_label: str, frac: float = 0.3) -> VoIResult:
    """Compute EVPPI and probability of decision change for one input.

    Implements Straub et al. (2025) Eq. 30: S is used only to pick the
    conditionally-optimal decision per sample; the raw utility (not the
    smoothed value) is used to evaluate both terms of the EVPPI.
    """
    x = samples.X_e[input_label]
    utilities = compute_utilities(samples.Y_e)  # shape (N, n_decisions), raw
    N, n_decisions = utilities.shape

    smoothed = np.empty((N, n_decisions))
    for d in range(n_decisions):
        smoothed[:, d] = fit_smoothing_estimator(x, utilities[:, d], input_label, x_eval=x, frac=frac)

    # a_opt|X_i^(k) = argmax_d S(x_i^(k), d) - smoothing picks the decision only
    optimal_decisions_perfect_info = np.argmax(smoothed, axis=1)

    # a_opt: status-quo decision, maximizing mean RAW utility across all samples
    optimal_decision_uncertain = int(np.argmax(utilities.mean(axis=0)))

    # Eq. 30: evaluate the RAW (unsmoothed) utility at each decision
    utility_perfect_info = utilities[np.arange(N), optimal_decisions_perfect_info]
    utility_uncertain = utilities[:, optimal_decision_uncertain]

    evppi = float(np.mean(utility_perfect_info) - np.mean(utility_uncertain))
    prob_change = float(np.mean(optimal_decisions_perfect_info != optimal_decision_uncertain))

    return VoIResult(
        input_label=input_label,
        evppi=evppi,
        prob_change=prob_change,
        optimal_decision_uncertain=optimal_decision_uncertain,
    )


def compute_all_evppi(samples: EpistemicSamples, frac: float = 0.3) -> list:
    """Compute VoIResult for every epistemic input, in cfg.X_E_LABELS order."""
    return [compute_evppi(samples, label, frac=frac) for label in cfg.X_E_LABELS]


def plot_smoothing_estimator(
    samples: EpistemicSamples,
    input_label: str,
    frac: float = 0.3,
    location_index=None,
    ax=None,
    decision_index=None,
):
    """Scatter of utility vs input, per decision, with S(x_i, d) overlaid.

    Mirrors Figure 1 / Figure 7 of Straub et al. (2025): raw samples as a
    scatter, the smoothing estimator as an overlay showing the fitted
    conditional expectation.

    Parameters
    ----------
    ax : None, Axes, or array of Axes, optional
        If None (default), creates a new figure with a 1×n_decisions
        horizontal layout and returns the figure.
        If a single Axes is provided and decision_index is specified,
        plots only that decision into the single axis and returns the axis.
        If a 1D array of Axes is provided and decision_index is None,
        plots all decisions into the corresponding axes.
    decision_index : int, optional
        If provided, plot only the specified decision (0-indexed) into
        the provided ax (single Axes). Ignored if ax is None.
    """
    x = samples.X_e[input_label]
    utilities = compute_utilities(samples.Y_e)
    n_decisions = utilities.shape[1]
    is_categorical = input_label in CATEGORICAL_INPUTS

    # Single-decision mode: plot only one decision into a single axis
    if decision_index is not None:
        if ax is None:
            raise ValueError("decision_index requires ax to be provided")
        d = decision_index
        ax.scatter(x, utilities[:, d], s=8, alpha=0.25, color=cfg.DECISION_COLORS[decision_index])

        if is_categorical:
            categories = np.unique(x)
            smoothed_vals = fit_smoothing_estimator(x, utilities[:, d], input_label, x_eval=categories)
            ax.scatter(categories, smoothed_vals, color="black", marker="D", s=70, zorder=5)
        else:
            order = np.argsort(x)
            x_sorted = np.asarray(x)[order]
            smoothed_vals = fit_smoothing_estimator(
                x, utilities[:, d], input_label, x_eval=x_sorted, frac=frac
            )
            ax.plot(x_sorted, smoothed_vals, color="black", linewidth=2)

        ax.set_xlabel(input_label)
        ax.set_ylabel("Utility")
        ax.set_title(cfg.DECISION_LABELS[d])
        # ax.legend(fontsize=8)
        if is_categorical:
            ax.tick_params(axis="x")
        return ax

    # Multi-decision mode: plot all decisions
    if ax is None:
        # Default mode: create new figure with horizontal layout
        fig, axes = plt.subplots(1, n_decisions, figsize=(5 * n_decisions, 4), sharex=True, sharey=True)
        if n_decisions == 1:
            axes = [axes]
        own_figure = True
    else:
        # Embedding mode: use provided axes
        ax_arr = np.atleast_1d(ax)
        if len(ax_arr) != n_decisions:
            raise ValueError(f"Expected {n_decisions} axes, got {len(ax_arr)}")
        axes = ax_arr
        own_figure = False

    for d in range(n_decisions):
        ax_d = axes[d]
        ax_d.scatter(x, utilities[:, d], s=8, alpha=0.25, color=cfg.DECISION_COLORS[d])

        if is_categorical:
            categories = np.unique(x)
            smoothed_vals = fit_smoothing_estimator(x, utilities[:, d], input_label, x_eval=categories)
            ax_d.scatter(categories, smoothed_vals, color="black", marker="D", s=70, zorder=5)
        else:
            order = np.argsort(x)
            x_sorted = np.asarray(x)[order]
            smoothed_vals = fit_smoothing_estimator(
                x, utilities[:, d], input_label, x_eval=x_sorted, frac=frac
            )
            ax_d.plot(x_sorted, smoothed_vals, color="black", linewidth=2)

        ax_d.set_xlabel(input_label)
        if d == 0:
            ax_d.set_ylabel("Utility")
        ax_d.set_title(cfg.DECISION_LABELS[d])
        # ax_d.legend(fontsize=8)
        if is_categorical:
            ax_d.tick_params(axis="x")

    if own_figure:
        title = f"Smoothing estimator for {input_label}"
        if location_index is not None:
            title += f" (location {location_index})"
        fig.suptitle(title)
        fig.tight_layout()
        return fig
    else:
        return None