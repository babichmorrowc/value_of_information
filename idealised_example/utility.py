"""
Shared utility function: converts Y_e(d) loss values into utilities

Original:
Combine the financial term (scaled by a fixed max-loss normalising
constant) and the non-financial alignment score for each decision:

    u(Y_e(d), d) = financial_weight * (1 - Y_e(d) / max_loss)
                 + nonfinancial_weight * (s_d / max_obj_score)

New:
Financial utility only: u(Y_e(d), d) = -Y_e(d).
"""
import numpy as np

import config as cfg

# Old utility function weighting financial and non-financial:
# def compute_utilities(losses, max_loss: float) -> np.ndarray:
#     """Compute u(Y_e(d), d) for one or many samples.

#     losses : array-like, shape (n_decisions,) or (N, n_decisions)
#         Y_e(d) values. cfg.OBJ_SCORES is broadcast against the last axis,
#         so both a single sample's losses and a full (N, n_decisions)
#         matrix work.
#     max_loss : float
#         The fixed normalising constant Y_max - the maximum financial loss
#         across all locations, decisions, and values of X (see the methods
#         chapter). Passed explicitly rather than defaulted, since it's a
#         precomputed value specific to a given run of the analysis.

#     Returns an array the same shape as `losses`.
#     """
#     losses = np.asarray(losses, dtype=float)
#     obj_scores = np.asarray(cfg.OBJ_SCORES, dtype=float)
#     fin_weight, nonfin_weight = cfg.UTILITY_WEIGHTS

#     financial_util = 1.0 - (losses / max_loss)
#     nonfinancial_util = obj_scores / cfg.MAX_OBJ_SCORE

#     return fin_weight * financial_util + nonfin_weight * nonfinancial_util

def compute_utilities(losses) -> np.ndarray:
    """Compute u(Y_e(d), d) = -Y_e(d) for one or many samples.
 
    losses : array-like, shape (n_decisions,) or (N, n_decisions)
        Y_e(d) values, in raw GBP.
 
    Returns an array the same shape as `losses`. argmax(utility) is
    exactly argmin(losses).
    """
    losses = np.asarray(losses, dtype=float)
    return -losses
