"""Configuration and constants for the heat-stress VoI/PAWN sensitivity analysis.

Both PAWN and VoI are applied to the epistemic inputs X_e only (theta, the
aleatory expected annual impact, is marginalised out via the GAM Monte Carlo
average and does not appear as an input to either method). X_E_LABELS below
fixes the canonical order used everywhere X_e is represented as an array
(e.g. for PAWN's input matrix).
"""
from pathlib import Path

# ----- Paths -----
DATA_DIR = Path("/home/aw23877/Documents/bda_sensitivity_paper/bda_risk_dec_sensitivity/data/")
RESULTS_DIR = Path("./results")
FIGURES_DIR = Path("./figures")

# ----- Decisions -----
N_DECISIONS = 3
DECISION_LABELS = ["d1_no_action", "d2_modify_hours", "d3_cooling_equipment"]

# Non-financial alignment scores for each decision (d1, d2, d3).
# Higher score = better alignment with organisational objectives.
OBJ_SCORES = [5, 6, 4]

# Utility function weights: (financial_weight, nonfinancial_weight)
UTILITY_WEIGHTS = (0.8, 0.2)
MAX_OBJ_SCORE = 10.0

# ----- Risk (epistemic) input options -----
CALIBRATION_OPTS = ["UKCP_raw", "UKCP_BC", "ChangeFactor"]
WARMING_OPTS = ["2deg", "4deg"]
SSP_OPTS = ["1", "2", "5"]
VULN1_OPTS = ["53.78", "54.5", "55.79"]
VULN2_OPTS = ["-4.597", "-4.1", "-3.804"]

# Warming level -> SSP data year to use (matches the mapping hardcoded
# inside location_funcs.get_EAI_Exp_bundle). Kept here as the single
# source of truth for anywhere else this mapping is needed, e.g. spatial.py.
WARMING_LEVEL_TO_SSP_YEAR = {"2deg": 2041, "4deg": 2084}

# ----- Decision (epistemic) input ranges -----
DC_RANGE = (100, 300)  # cost per day of work lost, shared across all decisions

# Per-decision ranges for annual cost per person (AC) and efficacy (E).
# d1 ("no action") has no cost/efficacy to sample - it is fixed at zero
# directly in sampling.compute_Ye_matrix and does not appear in X_E_LABELS.
AC_RANGES = {1: (150, 350), 2: (500, 700)}  # keyed by zero-indexed decision (d2, d3)
E_RANGES = {1: (0.3, 0.5), 2: (0.7, 0.9)}

# ----- Canonical order of epistemic inputs for PAWN / VoI -----
X_E_LABELS = [
    "Calibration method",
    "Warming level",
    "SSP",
    "Vulnerability parameter 1",
    "Vulnerability parameter 2",
    "Cost per day of work lost",
    "Annual cost per person of d2",
    "Effectiveness of d2",
    "Annual cost per person of d3",
    "Effectiveness of d3",
]