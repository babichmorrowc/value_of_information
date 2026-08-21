import matplotlib.pyplot as plt

import config
import voi
from precompute_samples import load_precomputed
from sampling import EpistemicSamples

# ---- Import samples ----
samples = load_precomputed()

# ---- Run in a specific location ----
loc_index = 1058

samples_for_location = {
    "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
    "Y_e": samples['Y_e_all'][loc_index],
    "location_index": loc_index,
    "n_samples": samples["n_samples"],
}
samples_for_location = EpistemicSamples(**samples_for_location)

# # One input at a location:
# result = voi.compute_evppi(samples_for_location, "Cost per day of work lost")
# print(result.evppi, result.prob_change)

# Every input at a location:
results = voi.compute_all_evppi(samples_for_location)
results

# The paper-style scatter + smoothing-curve figure:
fig = voi.plot_smoothing_estimator(samples_for_location, "Warming level", location_index=loc_index)
plt.show()

fig = voi.plot_smoothing_estimator(samples_for_location, "Effectiveness of d2", location_index=loc_index)
plt.show()