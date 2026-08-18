"""
Quick manual check: generate X_e / Y_e(d) samples for a single location
using the real data, and print a summary to sanity-check the output.
"""
import numpy as np
import matplotlib.pyplot as plt
import time

import config as cfg
import spatial
import sampling


def summarize_location(location_index: int, n_samples: int = 5000, seed: int = 0):
    """Generate samples for one location and print a summary.

    Note: the "naive decision" counts below use argmin(Y_e) - i.e. lowest
    financial loss only, ignoring the non-financial score weighting. That
    weighting is applied later, in the PAWN/VoI modules - this is purely a
    check on sampling.py's own output (X_e, Y_e), not the final decision
    rule.
    """
    grid = spatial.load_spatial_grid()
    rng = np.random.default_rng(seed)

    samples = sampling.generate_location_samples(
        location_index=location_index,
        ind=grid.ind,
        n_samples=n_samples,
        rng=rng,
    )

    print(f"Location index: {location_index}")
    print(f"Lat/lon: {grid.lat[location_index]:.3f}, {grid.lon[location_index]:.3f}")
    print(f"X_e keys: {list(samples.X_e.keys())}")
    print(f"Y_e shape: {samples.Y_e.shape}")

    print("\nExpected loss (mean Y_e) per decision:")
    for d in range(cfg.N_DECISIONS):
        mean_loss = samples.Y_e[:, d].mean()
        std_loss = samples.Y_e[:, d].std()
        print(f"  {cfg.DECISION_LABELS[d]}: mean {mean_loss:,.0f}  (std {std_loss:,.0f})")

    naive_decision = np.argmin(samples.Y_e, axis=1)
    counts = np.bincount(naive_decision, minlength=cfg.N_DECISIONS)
    print("\nNaive (financial-only) decision counts:")
    for d in range(cfg.N_DECISIONS):
        print(f"  {cfg.DECISION_LABELS[d]}: {counts[d]} ({100 * counts[d] / n_samples:.1f}%)")

    if counts.min() == 0:
        print(
            "\n  NB: at least one decision was never optimal for this location/sample "
            "- worth checking that's expected (e.g. very low or very high risk location) "
            "rather than a bug."
        )

    return samples


def plot_Ye_distribution(samples: sampling.EpistemicSamples, location_index: int):
    """Histogram of Y_e(d) per decision, for a quick visual check."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in range(cfg.N_DECISIONS):
        ax.hist(samples.Y_e[:, d], bins=30, alpha=0.5, label=cfg.DECISION_LABELS[d])
        ax.axvline(samples.Y_e[:, d].mean(), linestyle="--", color=f"C{d}")
    ax.set_xlabel("Y_e(d)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Y_e(d) - location {location_index}")
    ax.legend()
    return fig

def map_optimal_decisions(
    n_samples: int = 2000,
    seed: int = None,
    location_indices=None,
    progress_every: int = 100,
):
    """Compute and map the optimal decision under uncertainty at every location.
 
    Optimal decision here means argmin_d E[Y_e(d)] - same rule as
    check_optimal_decision.py, just run across many locations and plotted
    spatially instead of printed for one.
 
    location_indices : optional subset of location indices to run (e.g.
        range(200) for a quick test) - defaults to every location in the
        spatial grid. NB this loop calls get_EAI_Exp_bundle once per
        location (inside sampling.compute_Ye_matrix), which reads a NetCDF
        file per risk-input combination - this is the same cost as your
        original script's max_loss sweep, and can be slow over the full
        ~1700 locations. Try a subset first.
 
    Uses one shared X_e sample across all locations (see the note in
    sampling.generate_location_samples on why that's preferred over
    resampling per location).
    """
    grid = spatial.load_spatial_grid()
    rng = np.random.default_rng(seed if seed is not None else cfg.RANDOM_SEED)
    X_e = sampling.sample_epistemic_inputs(n_samples, rng)
 
    if location_indices is None:
        location_indices = range(len(grid.lat))
    location_indices = list(location_indices)
 
    optimal_decisions = np.full(len(location_indices), -1, dtype=int)
    for i, loc_ind in enumerate(location_indices):
        samples = sampling.generate_location_samples(
            location_index=loc_ind, ind=grid.ind, n_samples=n_samples, rng=None, X_e=X_e
        )
        expected_losses = samples.Y_e.mean(axis=0)
        optimal_decisions[i] = int(np.argmin(expected_losses))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{len(location_indices)} locations done...")
 
    lat = grid.lat[location_indices]
    lon = grid.lon[location_indices]
 
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    for d in range(cfg.N_DECISIONS):
        mask = optimal_decisions == d
        ax.scatter(lon[mask], lat[mask], s=12, label=cfg.DECISION_LABELS[d])
    ax.coastlines(color="grey", linewidth=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Optimal decision under uncertainty (argmin E[Y_e])")
    ax.legend()
 
    return fig, optimal_decisions


# London:
# timer_start = time.time()
samples_lon = summarize_location(location_index=241, n_samples=5000)
# timer_end = time.time()
# print(f'Time taken to summarize_location: {(timer_end - timer_start) / 60:.2f} minutes')
# 0.07 minutes with 5000 samples
fig = plot_Ye_distribution(samples_lon, location_index=241)
plt.show()

# Lake District:
samples_ld = summarize_location(location_index=1058, n_samples=5000)
fig = plot_Ye_distribution(samples_ld, location_index=1058)
plt.show()

# Scotland:
samples_sc = summarize_location(location_index=1460, n_samples=5000)
fig = plot_Ye_distribution(samples_sc, location_index=1460)
plt.show()

# Get samples for all locations in the grid
# Plot a map of the percentage of samples where each decision is optimal, for a quick sanity check
# Store the percentages for each location and each decision in a 2D array for plotting
percentages_array = np.zeros((1711, cfg.N_DECISIONS))
for loc in range(1711):
    print(f"Processing location {loc}...")
    samples = summarize_location(location_index=loc, n_samples=500)
    naive_decision = np.argmin(samples.Y_e, axis=1)
    counts = np.bincount(naive_decision, minlength=cfg.N_DECISIONS)
    percentages = 100 * counts / 500
    percentages_array[loc] = percentages
    print(f"Location {loc}: {percentages}")

# Plot the percentages for each decision on a map
from cartopy import crs as ccrs

grid = spatial.load_spatial_grid()
fig = plt.figure(figsize=(12, 8))
for d in range(cfg.N_DECISIONS):
    ax = fig.add_subplot(2, 3, d + 1, projection=ccrs.PlateCarree())
    sc = ax.scatter(grid.lon, grid.lat, c=percentages_array[:, d], cmap="Greens", s=10, vmin=0, vmax=100)
    ax.set_title(f"Decision {d + 1}: % optimal")
    ax.coastlines()
    plt.colorbar(sc, ax=ax, orientation="vertical", label="% optimal")
plt.tight_layout()
plt.show()

# Map the optimal decision under uncertainty for a subsample of locations:
timer_start = time.time()
fig, optimal_decisions = map_optimal_decisions(n_samples=2000, location_indices=range(1500,1700), progress_every=1)
timer_end = time.time()
print(f"Time: {(timer_end - timer_start) / 60:.2f} minutes")
# 12.65 minutes for 200 locations and 2000 samples/location
plt.show()

# Map the optimal decision under uncertainty for all locations:
timer_start = time.time()
fig, optimal_decisions = map_optimal_decisions(n_samples=200, progress_every=1)
timer_end = time.time()
print(f"Time: {(timer_end - timer_start) / 60:.2f} minutes")
# -- minutes for all locations and 200 samples/location
plt.show()