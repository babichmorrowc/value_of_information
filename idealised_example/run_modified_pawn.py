"""
Generate the shared X_e / Y_e(d) samples for a location and run the
modified-PAWN sensitivity analysis on them.

Run from the directory containing config.py, spatial.py, sampling.py,
utility.py, modified_pawn.py, your real location_funcs.py.
"""
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
import spatial
import sampling
import modified_pawn


def run_pawn_for_location(
    location_index: int,
    grid: spatial.SpatialGrid,
    X_e: dict,
    n_samples: int,
) -> modified_pawn.PawnResult:
    """Compute modified-PAWN indices for one location, using a shared X_e sample."""
    samples = sampling.generate_location_samples(
        location_index=location_index,
        ind=grid.ind,
        n_samples=n_samples,
        rng=None,  # unused when X_e is provided
        X_e=X_e,
    )
    return modified_pawn.compute_pawn_indices(samples)


def print_pawn_result(result: modified_pawn.PawnResult, location_index: int):
    print(f"\nmodified-PAWN sensitivity indices - location {location_index}")
    if result.degenerate:
        print("  (degenerate: every sample picked the same decision - sensitivity undefined)")
        return
    # Sort inputs by sensitivity, most sensitive first
    order = np.argsort(result.max_dist_mean)[::-1]
    for i in order:
        print(
            f"  {result.labels[i]:35s} "
            f"S_i = {result.max_dist_mean[i]:.3f} "
            f"[{result.max_dist_lb[i]:.3f}, {result.max_dist_ub[i]:.3f}]"
        )

def plot_pawn_bargraph(result: modified_pawn.PawnResult, location_index: int):
    original_errors = np.array([result.max_dist_lb, result.max_dist_ub])
    error_bars = np.zeros_like(original_errors)
    error_bars[0,:] = result.max_dist_mean - original_errors[0,:]
    error_bars[1,:] = original_errors[1,:] - result.max_dist_mean
    fig = plt.figure(figsize=(15,10))
    plt.bar(result.labels, result.max_dist_mean, yerr=error_bars)
    plt.title(f'Modified PAWN sensitivity index for location {location_index}')
    return fig

# Draw X_e once, up front, and reuse it for every location - see the
# note on shared sampling in sampling.generate_location_samples.
n_samples = 5000
rng = np.random.default_rng(cfg.RANDOM_SEED)
grid = spatial.load_spatial_grid()
X_e = sampling.sample_epistemic_inputs(n_samples, rng)
# London and the Lake District, per your original script's indices -
# swap in / extend this list with whatever locations you want.
location_indices = {"London": 241, "Lake District": 1058}
for name, loc_ind in location_indices.items():
    result = run_pawn_for_location(location_index=loc_ind,
                                   grid=grid,
                                   X_e=X_e,
                                   n_samples=n_samples)
    print_pawn_result(result, location_index=loc_ind)
    # plot_pawn_result(result, location_index=loc_ind)

plot_pawn_bargraph(result, loc_ind)
plt.show()