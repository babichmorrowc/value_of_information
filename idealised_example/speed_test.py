# Compare the speed of modified PAWN to VoI in one location
import time
import precompute_samples as pc
import modified_pawn
import voi

# Import the precomputed samples
cache = pc.load_precomputed("samples_cache.npz")

# Try both out in London
loc_index = 241

# Try PAWN out in London to time
start_time = time.time()
samples = pc.get_location_samples(cache, location_index=int(loc_index))
pawn_result = modified_pawn.compute_pawn_indices(samples)
pawn_time = time.time() - start_time
print(f"Modified PAWN took {pawn_time:.3f} seconds for location {loc_index}")

# Try VoI out in London to time
start_time = time.time()
voi_result = voi.compute_all_evppi(samples)
voi_time = time.time() - start_time
print(f"VoI took {voi_time:.3f} seconds for location {loc_index}")
