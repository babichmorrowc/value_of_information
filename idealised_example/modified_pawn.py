"""
Apply the modified PAWN (PAWN on PMFs) sensitivity method to the shared
X_e / Y_e(d) sample base produced by sampling.py.

PAWN's decision output Y is d = argmax_d u(Y_e(d), d) - the same
epistemically-conditioned decision VoI's d_opt is built from (see
sampling.py's module docstring and the methods chapter for the
rationale for restricting PAWN to X_e).

NOTE: safepython.PAWN_pmf is NOT part of the released `safepython` pip
package (only PAWN, VBSA, RSA_groups, etc. are) - it's custom to your
fork (github.com/babichmorrowc/SAFE-python). Make sure that fork's
PAWN_pmf.py is importable as safepython.PAWN_pmf in your environment
(e.g. `pip install git+https://github.com/babichmorrowc/SAFE-python.git`,
or drop PAWN_pmf.py into your installed safepython package directory).
Everything else this module needs (pawn_split_sample, pawn_ks, allrange,
aggregate_boot) is already in the standard pip package.
"""
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from safepython.PAWN_pmf import pawn_pmf_indices
from safepython.util import aggregate_boot

import config as cfg
from sampling import EpistemicSamples
from utility import compute_utilities


@dataclass
class PawnResult:
    """Modified-PAWN sensitivity results for one location.

    labels : list[str]
        Input labels, in the same order as the other arrays (matches
        cfg.X_E_LABELS).
    max_dist_mean : np.ndarray, shape (M,)
        Point estimate (mean across bootstrap resamples) of the
        sensitivity index S_i for each input - i.e. the mean maximum
        vertical distance between conditional and unconditional PMFs,
        per the methods chapter's Equation for S_i.
    max_dist_lb, max_dist_ub : np.ndarray, shape (M,)
        Bootstrap confidence interval bounds for max_dist_mean.
    degenerate : bool
        True if every sample at this location picked the same decision,
        making the sensitivity index undefined (all arrays are NaN in
        this case) rather than zero.
    """
    labels: list
    max_dist_mean: np.ndarray
    max_dist_lb: np.ndarray
    max_dist_ub: np.ndarray
    degenerate: bool = False


def encode_Xe_numeric(X_e: dict) -> np.ndarray:
    """Encode X_e as a numeric (N, M) array, in cfg.X_E_LABELS order.

    pawn_pmf_indices requires numeric input. Categorical risk inputs are
    mapped as follows (matching the encoding used in the original
    sensitivity_analysis.py):
      - Warming level -> the actual degree value (parsed from "2deg"/"4deg")
      - SSP -> the actual SSP number
      - Vulnerability parameters 1 & 2 -> their actual float values
      - Calibration method -> an arbitrary ordinal code (0, 1, 2), since it
        has no natural numeric scale
    Decision inputs (CD, AC_d, E_d) are already numeric and used as-is.
    """
    n_samples = len(X_e["Cost per day of work lost"])

    calibration_codes = np.array(
        [cfg.CALIBRATION_OPTS.index(v) for v in X_e["Calibration method"]], dtype=float
    )
    warming_numeric = np.array(
        [float(v.replace("deg", "")) for v in X_e["Warming level"]], dtype=float
    )
    ssp_numeric = np.array([float(v) for v in X_e["SSP"]], dtype=float)
    vuln1_numeric = np.array([float(v) for v in X_e["Vulnerability parameter 1"]], dtype=float)
    vuln2_numeric = np.array([float(v) for v in X_e["Vulnerability parameter 2"]], dtype=float)

    columns = {
        "Calibration method": calibration_codes,
        "Warming level": warming_numeric,
        "SSP": ssp_numeric,
        "Vulnerability parameter 1": vuln1_numeric,
        "Vulnerability parameter 2": vuln2_numeric,
        "Cost per day of work lost": np.asarray(X_e["Cost per day of work lost"], dtype=float),
        "Annual cost per person of d2": np.asarray(X_e["Annual cost per person of d2"], dtype=float),
        "Effectiveness of d2": np.asarray(X_e["Effectiveness of d2"], dtype=float),
        "Annual cost per person of d3": np.asarray(X_e["Annual cost per person of d3"], dtype=float),
        "Effectiveness of d3": np.asarray(X_e["Effectiveness of d3"], dtype=float),
    }

    X_numeric = np.empty((n_samples, len(cfg.X_E_LABELS)))
    for j, label in enumerate(cfg.X_E_LABELS):
        X_numeric[:, j] = columns[label]
    return X_numeric


def compute_pawn_indices(
    samples: EpistemicSamples,
    n_conditioning_intervals: int = cfg.PAWN_N_CONDITIONING_INTERVALS,
    n_bootstrap: int = cfg.PAWN_N_BOOTSTRAP,
) -> PawnResult:
    """Compute the modified-PAWN sensitivity index for every epistemic input.

    samples.Y_e is converted to a per-sample optimal decision (via
    utility.compute_utilities + argmax) before being passed to
    pawn_pmf_indices as the discrete output Y - this is PAWN's
    d = argmax_d u(Y_e(d), d), restricted to X_e (see module docstring).
    """
    X_numeric = encode_Xe_numeric(samples.X_e)

    utilities = compute_utilities(samples.Y_e)  # shape (N, n_decisions)
    Y = np.argmax(utilities, axis=1).astype(float)  # pawn_pmf_indices expects numeric Y

    if len(np.unique(Y)) == 1:
        # Every sample picked the same decision at this location - the
        # conditional and unconditional PMFs would be identical and
        # degenerate (all mass on one value), so sensitivity here is
        # undefined rather than meaningfully zero.
        n_inputs = len(cfg.X_E_LABELS)
        nan_arr = np.full(n_inputs, np.nan)
        return PawnResult(
            labels=list(cfg.X_E_LABELS),
            max_dist_mean=nan_arr,
            max_dist_lb=nan_arr,
            max_dist_ub=nan_arr,
            degenerate=True,
        )

    # max_dist_mean_boot: shape (Nboot, M) - the "mean across conditioning
    # intervals" statistic, per bootstrap resample. The methods chapter
    # specifies the mean as the summary statistic for S_i (median and max
    # are also returned by pawn_pmf_indices but not used here).
    _, max_dist_mean_boot, _ = pawn_pmf_indices(
        X_numeric, Y, n=n_conditioning_intervals, Nboot=n_bootstrap
    )
    mean_across_boot, lb, ub = aggregate_boot(max_dist_mean_boot)

    return PawnResult(
        labels=list(cfg.X_E_LABELS),
        max_dist_mean=mean_across_boot,
        max_dist_lb=lb,
        max_dist_ub=ub,
    )

def plot_pawn_bargraph(result: PawnResult, location_index: int, ax=None, **kwargs):
    original_errors = np.array([result.max_dist_lb, result.max_dist_ub])
    error_bars = np.zeros_like(original_errors)
    error_bars[0, :] = result.max_dist_mean - original_errors[0, :]
    error_bars[1, :] = original_errors[1, :] - result.max_dist_mean

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    else:
        fig = ax.figure

    ax.bar(result.labels, result.max_dist_mean, yerr=error_bars, **kwargs)
    ax.set_title(f"Modified PAWN sensitivity index for location {location_index}")
    ax.set_xlabel("Input")
    ax.set_ylabel("Sensitivity index")
    ax.tick_params(axis="x", rotation=45)
    return ax