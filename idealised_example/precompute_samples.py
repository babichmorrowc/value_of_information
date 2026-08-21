"""
Precompute the shared X_e sample and every location's Y_e(d), and cache
the result to a single .npz file.

This is the expensive step - each location's Y_e(d) requires reading a
NetCDF file per risk-input combination (via get_EAI_Exp_bundle, inside
sampling.compute_Ye_matrix). Running it once and caching the result lets
PAWN and (eventually) VoI reload the same samples and iterate on their
own settings (bootstrap count, plots, etc.) without re-paying that cost.

Run from the directory containing config.py, spatial.py, sampling.py, and
your real location_funcs.py.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
import spatial
import sampling
# from location_check import plot_Ye_distribution


def precompute_all_locations(
    n_samples: int,
    location_indices=None,
    seed: int = None,
    out_path: str = "samples_cache.npz",
    progress_every: int = 100,
):
    """Draw X_e once, compute Y_e(d) for each location, and save to out_path.

    location_indices : optional subset (e.g. [241, 1058] for just London
        and the Lake District) - defaults to every location in the
        spatial grid. Start with a subset while iterating; run the full
        set once you're ready to commit to the analysis.
    """
    grid = spatial.load_spatial_grid()
    rng = np.random.default_rng(seed if seed is not None else cfg.RANDOM_SEED)
    X_e = sampling.sample_epistemic_inputs(n_samples, rng)

    if location_indices is None:
        location_indices = range(len(grid.lat))
    location_indices = np.array(list(location_indices))

    Y_e_all = np.empty((len(location_indices), n_samples, cfg.N_DECISIONS))
    for i, loc_ind in enumerate(location_indices):
        samples = sampling.generate_location_samples(
            location_index=int(loc_ind), ind=grid.ind, n_samples=n_samples, rng=None, X_e=X_e
        )
        Y_e_all[i] = samples.Y_e
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{len(location_indices)} locations done...")

    save_kwargs = {f"X_e__{k}": v for k, v in X_e.items()}
    np.savez(
        out_path,
        location_indices=location_indices,
        lat=grid.lat[location_indices],
        lon=grid.lon[location_indices],
        Y_e_all=Y_e_all,
        n_samples=n_samples,
        **save_kwargs,
    )
    print(f"Saved {len(location_indices)} locations x {n_samples} samples to {out_path}")
    return X_e, Y_e_all, location_indices


def load_precomputed(path: str = "samples_cache.npz") -> dict:
    """Load a cache written by precompute_all_locations.

    Returns a dict with keys: X_e (dict), Y_e_all (array, shape
    (n_locations, n_samples, n_decisions)), location_indices, lat, lon,
    n_samples.
    """
    data = np.load(path, allow_pickle=False)
    prefix = "X_e__"
    X_e = {key[len(prefix):]: data[key] for key in data.files if key.startswith(prefix)}
    return {
        "X_e": X_e,
        "Y_e_all": data["Y_e_all"],
        "location_indices": data["location_indices"],
        "lat": data["lat"],
        "lon": data["lon"],
        "n_samples": int(data["n_samples"]),
    }


def get_location_samples(cache: dict, location_index: int) -> sampling.EpistemicSamples:
    """Reconstruct one location's EpistemicSamples from a loaded cache."""
    matches = np.where(cache["location_indices"] == location_index)[0]
    if len(matches) == 0:
        raise KeyError(f"Location {location_index} not found in this cache.")
    i = matches[0]
    return sampling.EpistemicSamples(
        X_e=cache["X_e"],
        Y_e=cache["Y_e_all"][i],
        location_index=location_index,
        n_samples=cache["n_samples"],
    )

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

def plot_cached_location(location_index: int, cache_path: str = "samples_cache.npz"):
    """Load the cache, pull out one location's samples, and plot Y_e(d)."""
    cache = load_precomputed(cache_path)
    samples = get_location_samples(cache, location_index)
    fig = plot_Ye_distribution(samples, location_index)
    return fig



if __name__ == "__main__":
    # Small example: cache just London and the Lake District while testing.
    # Drop location_indices (or pass None) to run the full spatial grid.
    precompute_all_locations(
        n_samples=10000,
        # location_indices=[1445, 1449],
        out_path="samples_cache.npz",
    )

