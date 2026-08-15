"""
Quick manual check: generate X_e / Y_e(d) samples for a single location
using the real data, and print a summary to sanity-check the output.

Run this from the directory containing config.py, spatial.py, sampling.py,
and your real location_funcs.py. Uses config.DATA_DIR - double check that
still points at the right place in whatever environment you're running
this in.
"""
import numpy as np
import matplotlib.pyplot as plt

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

# London, per your original script's location index - swap in whatever
# location you want to check.
samples = summarize_location(location_index=241, n_samples=5000)
fig = plot_Ye_distribution(samples, location_index=241)
plt.show()