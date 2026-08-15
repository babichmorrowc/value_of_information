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

NOTE: get_EAI_Exp_bundle is imported from location_funcs, treated here as a
fixed black box. calc_Ye_jit is defined in this module (moved from
location_funcs) since it's the core per-sample loss calculation the rest of
sampling.py is built around. This module has not been executed against the
real data/location_funcs in this environment - please run it against your
actual location_funcs.py before relying on it, in case the real function
signatures differ from what's assumed here (matched against the original
script you shared).
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
    # Average cost over the 1000 GAM samples
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

    return Y_e


def generate_location_samples(
    location_index: int,
    ind: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> EpistemicSamples:
    """Generate the full shared sample base for one location.

    This is the single entry point both PAWN and VoI build on: draw X_e
    once, compute Y_e(d) once, and hand both back for the two methods to
    consume independently downstream.
    """
    X_e = sample_epistemic_inputs(n_samples, rng)
    Y_e = compute_Ye_matrix(location_index, X_e, ind)
    return EpistemicSamples(
        X_e=X_e,
        Y_e=Y_e,
        location_index=location_index,
        n_samples=n_samples,
    )