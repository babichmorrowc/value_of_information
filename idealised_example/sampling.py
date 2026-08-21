"""
Generate shared samples of the epistemic inputs X_e and the corresponding
Y_e(d) loss values, for use by both the PAWN and VoI sensitivity methods.

Both methods are built on the same underlying quantity in this application:
Y_e(d), the epistemically-conditioned expected loss (aleatory theta already
marginalised out via the GAM Monte Carlo average - see Algorithm 1 in the
methods chapter). Generating X_e and Y_e(d) once per location and reusing
them for both PAWN and VoI is an efficiency choice, not a methodological
shortcut: the per-sample decision d = argmax_d u(Y_e(d), d) is exactly the
quantity PAWN's sensitivity index is computed over, and the same Y_e(d)
matrix is what VoI's EVPPI calculation consumes directly.
"""
from dataclasses import dataclass

import numpy as np
from numba import jit

import config as cfg
from location_funcs import get_EAI_Exp_bundle


@jit(nopython=True)
def calc_Ye_jit(EAI_Exp, decision_inputs):
    """Calculate Y_e(d) for one sample at one location, for one decision.

    EAI_Exp : tuple (EAI_samples, ppl)
        EAI_samples is the array of 1000 GAM samples of expected annual
        impact for this location/risk-input combination; ppl is the
        exposed population (number of people in outdoor physical work)
        at this location.
    decision_inputs : sequence of 3 floats
        [DC, AC, E] - cost per day of work lost, annual cost per person
        of the decision, and the decision's efficacy.

    Returns the Monte Carlo average of financial loss over the 1000 GAM
    samples of theta (the aleatory expected annual impact), i.e. Y_e(d)
    marginalised over theta at this fixed set of epistemic inputs.
    """
    EAI_samples, ppl = EAI_Exp
    cost = np.empty(1000)
    for k in range(1000):  # loop over GAM samples of theta
        cost[k] = decision_inputs[1] * ppl + decision_inputs[0] * (1 - decision_inputs[2]) * EAI_samples[k]
    return np.mean(cost)


@dataclass
class EpistemicSamples:
    """Shared sample base for one location, consumed by both PAWN and VoI.

    Attributes
    ----------
    X_e : dict[str, np.ndarray]
        One entry per label in cfg.X_E_LABELS, each an array of length N.
        Risk inputs (calibration, warming, SSP, vuln1, vuln2) are arrays of
        strings (categorical); decision inputs (CD, AC_d, E_d) are arrays
        of floats.
    Y_e : np.ndarray
        Shape (N, cfg.N_DECISIONS). Y_e[i, d] is the epistemically-
        conditioned expected loss for sample i under decision d.
    location_index : int
    n_samples : int
    """
    X_e: dict
    Y_e: np.ndarray
    location_index: int
    n_samples: int


def _sample_risk_inputs(n_samples: int, rng: np.random.Generator) -> dict:
    """Draw N samples of each risk-related epistemic input (discrete uniform)."""
    return {
        "Calibration method": rng.choice(cfg.CALIBRATION_OPTS, size=n_samples),
        "Warming level": rng.choice(cfg.WARMING_OPTS, size=n_samples),
        "SSP": rng.choice(cfg.SSP_OPTS, size=n_samples),
        "Vulnerability parameter 1": rng.choice(cfg.VULN1_OPTS, size=n_samples),
        "Vulnerability parameter 2": rng.choice(cfg.VULN2_OPTS, size=n_samples),
    }


def _sample_decision_inputs(n_samples: int, rng: np.random.Generator) -> dict:
    """Draw N samples of the decision-related epistemic inputs (continuous uniform).

    CD (cost per day of work lost) is a property of the workforce, shared
    across all three decisions. AC and E are sampled only for d2 and d3;
    d1 ("no action") has no cost/efficacy to learn about, so it contributes
    no entries to X_e (handled as a fixed zero in compute_Ye_matrix).
    """
    dc_low, dc_high = cfg.DC_RANGE
    samples = {
        "Cost per day of work lost": rng.uniform(dc_low, dc_high, size=n_samples),
    }
    for d in (1, 2):  # zero-indexed d2, d3
        ac_low, ac_high = cfg.AC_RANGES[d]
        e_low, e_high = cfg.E_RANGES[d]
        samples[f"Annual cost per person of d{d + 1}"] = rng.uniform(ac_low, ac_high, size=n_samples)
        samples[f"Effectiveness of d{d + 1}"] = rng.uniform(e_low, e_high, size=n_samples)
    return samples


def sample_epistemic_inputs(n_samples: int, rng: np.random.Generator) -> dict:
    """Draw N samples of all epistemic inputs X_e, keyed by cfg.X_E_LABELS."""
    X_e = {}
    X_e.update(_sample_risk_inputs(n_samples, rng))
    X_e.update(_sample_decision_inputs(n_samples, rng))
    missing = set(cfg.X_E_LABELS) - set(X_e)
    if missing:
        raise ValueError(f"Missing samples for epistemic inputs: {missing}")
    return X_e


def compute_Ye_matrix(location_index: int, X_e: dict, ind: np.ndarray) -> np.ndarray:
    """Compute Y_e(d) for every sample and every decision at one location.

    Returns an (N, cfg.N_DECISIONS) array. Decision d1 ("no action") uses
    AC = E = 0 for every sample, since it has no decision-specific inputs.
    """
    n_samples = len(X_e["Cost per day of work lost"])

    EAI_Exp_samples = get_EAI_Exp_bundle(
        index=location_index,
        ind=ind,
        input_data_path=cfg.DATA_DIR,
        calibration_opts=cfg.CALIBRATION_OPTS,
        warming_level_opts=cfg.WARMING_OPTS,
        ssp_opts=cfg.SSP_OPTS,
        vuln_param_1_opts=cfg.VULN1_OPTS,
        vuln_param_2_opts=cfg.VULN2_OPTS,
    )

    calibration = X_e["Calibration method"]
    warming = X_e["Warming level"]
    ssp = X_e["SSP"]
    vuln1 = X_e["Vulnerability parameter 1"]
    vuln2 = X_e["Vulnerability parameter 2"]
    DC = X_e["Cost per day of work lost"]

    # Some (location, SSP, warming level) combinations have zero exposed
    # population (no one working outdoor physical jobs there under that
    # scenario) - ppl is the second element of each EAI_Exp_samples entry.
    # With zero people, there's no one for heat stress to affect, so every
    # decision should show zero loss, and the optimal decision should
    # always be d1 ("no action"), since paying for d2/d3 would be pure
    # waste. Precompute this mask once (it only depends on the risk
    # inputs, not on which decision we're evaluating).
    ppl_per_sample = np.array(
        [
            EAI_Exp_samples[(calibration[i], warming[i], ssp[i], vuln1[i], vuln2[i])][1]
            for i in range(n_samples)
        ]
    )
    zero_population = ppl_per_sample == 0


    Y_e = np.empty((n_samples, cfg.N_DECISIONS))
    for d in range(cfg.N_DECISIONS):
        if d == 0:
            AC = np.zeros(n_samples)
            E = np.zeros(n_samples)
        else:
            AC = X_e[f"Annual cost per person of d{d + 1}"]
            E = X_e[f"Effectiveness of d{d + 1}"]

        for i in range(n_samples):
            key = (calibration[i], warming[i], ssp[i], vuln1[i], vuln2[i])
            EAI_Exp = EAI_Exp_samples[key]
            Y_e[i, d] = calc_Ye_jit(EAI_Exp, [DC[i], AC[i], E[i]])

    # Zero-population override: force exactly zero loss for every decision
    # on these samples, regardless of what calc_Ye_jit computed (theta may
    # still be non-zero there due to upstream data/interpolation artifacts,
    # even though nobody is exposed). Since all three decisions then tie at
    # exactly 0.0, argmax(-Y_e) / argmin(Y_e) selects d1 via NumPy's
    # documented first-occurrence tie-breaking (d1 is index 0 in
    # cfg.DECISION_LABELS) - this is a stable, documented contract, not an
    # implementation detail, but see tests/test_zero_population.py for a
    # regression test that pins this behaviour explicitly in case
    # DECISION_LABELS' order ever changes.
    Y_e[zero_population, :] = 0.0


    return Y_e


def generate_location_samples(
    location_index: int,
    ind: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    X_e: dict = None,
) -> EpistemicSamples:
    """Generate the full shared sample base for one location.

    This is the single entry point both PAWN and VoI build on: draw X_e
    (or reuse a shared one - see below), compute Y_e(d) once, and hand
    both back for the two methods to consume independently downstream.

    X_e : optional pre-drawn epistemic input samples (e.g. from a single
        upfront call to sample_epistemic_inputs), to reuse identical input
        draws across multiple locations. There's no requirement to redraw
        X_e per location - the risk/decision inputs aren't location-specific,
        only Y_e(d) is (via that location's EAI_Exp_samples). Reusing the
        same X_e across locations also means differences in Y_e/PAWN/VoI
        results between locations reflect the location's risk profile
        rather than sampling noise (common random numbers). If X_e is
        given, `rng` and `n_samples` are only used to validate/label the
        result - n_samples must match len(X_e) or a ValueError is raised.
        If X_e is None (the default), a fresh sample is drawn using rng.
    """
    if X_e is None:
        X_e = sample_epistemic_inputs(n_samples, rng)
    else:
        actual_n = len(X_e["Cost per day of work lost"])
        if actual_n != n_samples:
            raise ValueError(
                f"n_samples ({n_samples}) does not match the length of the "
                f"provided X_e ({actual_n})."
            )

    Y_e = compute_Ye_matrix(location_index, X_e, ind)
    return EpistemicSamples(
        X_e=X_e,
        Y_e=Y_e,
        location_index=location_index,
        n_samples=n_samples,
    )