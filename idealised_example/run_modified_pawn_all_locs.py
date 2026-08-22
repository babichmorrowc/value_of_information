"""
Run the modified PAWN sensitivity analysis across every location in a
precomputed samples cache, and save the resulting S_i values (with
bootstrap CIs) to a single file for later mapping.

Run from the directory containing config.py, sampling.py,
precompute_samples.py, modified_pawn.py, and samples_cache.npz (built by
precompute_samples.py - re-run that first if Y_e's computation has
changed, e.g. after the zero-population fix).
"""
import numpy as np

import config as cfg
import precompute_samples as pc
import modified_pawn


def run_pawn_all_locations(
    cache_path: str = "samples_cache.npz",
    out_path: str = "pawn_results.npz",
    n_bootstrap: int = None,
    progress_every: int = 100,
):
    """Compute modified-PAWN indices for every location in the cache.

    n_bootstrap : override cfg.PAWN_N_BOOTSTRAP if given (e.g. a smaller
        value for a quick full-country test run before committing to the
        default bootstrap count everywhere).
    """
    cache = pc.load_precomputed(cache_path)
    location_indices = cache["location_indices"]
    n_locations = len(location_indices)
    n_inputs = len(cfg.X_E_LABELS)

    max_dist_mean = np.full((n_locations, n_inputs), np.nan)
    max_dist_lb = np.full((n_locations, n_inputs), np.nan)
    max_dist_ub = np.full((n_locations, n_inputs), np.nan)
    degenerate = np.zeros(n_locations, dtype=bool)

    kwargs = {}
    if n_bootstrap is not None:
        kwargs["n_bootstrap"] = n_bootstrap

    for i, loc_ind in enumerate(location_indices):
        samples = pc.get_location_samples(cache, location_index=int(loc_ind))
        result = modified_pawn.compute_pawn_indices(samples, **kwargs)

        max_dist_mean[i] = result.max_dist_mean
        max_dist_lb[i] = result.max_dist_lb
        max_dist_ub[i] = result.max_dist_ub
        degenerate[i] = result.degenerate

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{n_locations} locations done...")

    np.savez(
        out_path,
        location_indices=location_indices,
        lat=cache["lat"],
        lon=cache["lon"],
        labels=np.array(cfg.X_E_LABELS),
        max_dist_mean=max_dist_mean,
        max_dist_lb=max_dist_lb,
        max_dist_ub=max_dist_ub,
        degenerate=degenerate,
    )
    n_degenerate = int(degenerate.sum())
    print(
        f"Saved PAWN results for {n_locations} locations to {out_path} "
        f"({n_degenerate} degenerate, i.e. every sample picked the same decision)"
    )

    return {
        "location_indices": location_indices,
        "lat": cache["lat"],
        "lon": cache["lon"],
        "labels": list(cfg.X_E_LABELS),
        "max_dist_mean": max_dist_mean,
        "max_dist_lb": max_dist_lb,
        "max_dist_ub": max_dist_ub,
        "degenerate": degenerate,
    }


def load_pawn_results(path: str = "pawn_results.npz") -> dict:
    """Load results saved by run_pawn_all_locations."""
    data = np.load(path, allow_pickle=False)
    return {
        "location_indices": data["location_indices"],
        "lat": data["lat"],
        "lon": data["lon"],
        "labels": [str(s) for s in data["labels"]],
        "max_dist_mean": data["max_dist_mean"],
        "max_dist_lb": data["max_dist_lb"],
        "max_dist_ub": data["max_dist_ub"],
        "degenerate": data["degenerate"],
    }


if __name__ == "__main__":
    run_pawn_all_locations(progress_every=5)