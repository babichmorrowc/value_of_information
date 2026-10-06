"""
Run the value-of-information analysis (EVPPI + probability of decision
change) across every location in a precomputed samples cache, and save
the results to a single file for later mapping.

Run from the directory containing config.py, sampling.py,
precompute_samples.py, voi.py, and samples_cache.npz.

NB on runtime: each location fits a LOESS smoother per (continuous input,
decision) pair - 5 continuous inputs x 3 decisions = 15 LOESS fits per
location, on top of the 5 categorical inputs' (cheap) group means. LOESS
is roughly O(N^2) per fit without further tuning, so with thousands of
samples per location this can add up over ~1700 locations.
"""
import time

import numpy as np

import config as cfg
import precompute_samples as pc
import voi


def run_voi_all_locations(
    cache_path: str = "samples_cache.npz",
    out_path: str = "voi_results.npz",
    location_indices=None,
    frac: float = 0.3,
    progress_every: int = 50,
):
    """Compute EVPPI and probability of decision change for every input,
    at every location (or the given subset).

    location_indices : optional subset of location indices (must already
        be present in the cache) - e.g. [241, 1058] to test timing before
        running the full set. Defaults to every location in the cache.
    """
    cache = pc.load_precomputed(cache_path)
    all_location_indices = cache["location_indices"]

    if location_indices is None:
        location_indices = all_location_indices
    else:
        location_indices = np.array(location_indices)

    n_locations = len(location_indices)
    n_inputs = len(cfg.X_E_LABELS)

    evppi = np.full((n_locations, n_inputs), np.nan)
    prob_change = np.full((n_locations, n_inputs), np.nan)
    optimal_decision_uncertain = np.full(n_locations, -1, dtype=int)

    # lat/lon for just the requested subset, matching cache's location order
    loc_to_cache_row = {int(loc): i for i, loc in enumerate(all_location_indices)}
    subset_rows = [loc_to_cache_row[int(loc)] for loc in location_indices]
    lat = cache["lat"][subset_rows]
    lon = cache["lon"][subset_rows]

    start = time.time()
    for i, loc_ind in enumerate(location_indices):
        samples = pc.get_location_samples(cache, location_index=int(loc_ind))
        results = voi.compute_all_evppi(samples, frac=frac)  # list of VoIResult, cfg.X_E_LABELS order

        for j, result in enumerate(results):
            evppi[i, j] = result.evppi
            prob_change[i, j] = result.prob_change
        # optimal_decision_uncertain doesn't depend on X_i, so it's the
        # same across all n_inputs results - just take the first
        optimal_decision_uncertain[i] = results[0].optimal_decision_uncertain

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - start
            rate = elapsed / (i + 1)
            print(f"  {i + 1}/{n_locations} locations done ({rate:.2f}s/location so far)...")

    np.savez(
        out_path,
        location_indices=location_indices,
        lat=lat,
        lon=lon,
        labels=np.array(cfg.X_E_LABELS),
        evppi=evppi,
        prob_change=prob_change,
        optimal_decision_uncertain=optimal_decision_uncertain,
    )
    print(f"Saved VoI results for {n_locations} locations to {out_path}")

    return {
        "location_indices": location_indices,
        "lat": lat,
        "lon": lon,
        "labels": list(cfg.X_E_LABELS),
        "evppi": evppi,
        "prob_change": prob_change,
        "optimal_decision_uncertain": optimal_decision_uncertain,
    }


def load_voi_results(path: str = "voi_results.npz") -> dict:
    """Load results saved by run_voi_all_locations."""
    data = np.load(path, allow_pickle=False)
    return {
        "location_indices": data["location_indices"],
        "lat": data["lat"],
        "lon": data["lon"],
        "labels": [str(s) for s in data["labels"]],
        "evppi": data["evppi"],
        "prob_change": data["prob_change"],
        "optimal_decision_uncertain": data["optimal_decision_uncertain"],
    }


if __name__ == "__main__":
    run_voi_all_locations()