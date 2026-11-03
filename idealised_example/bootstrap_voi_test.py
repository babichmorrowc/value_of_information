# Test out the new bootstrapped EVPPI computation
import numpy as np
import time

import config
import voi
from utility import compute_utilities
from precompute_samples import load_precomputed
from sampling import EpistemicSamples

# Read in samples
samples = load_precomputed()

# Test on one location index
loc = 1449  # any location index
s = EpistemicSamples(
    X_e={k: samples["X_e"][k] for k in config.X_E_LABELS},
    Y_e=samples["Y_e_all"][loc],
    location_index=loc,
    n_samples=samples["n_samples"],
)

t = time.time(); voi.compute_evppi_bootstrap(s, "Cost per day of work lost", n_boot=10); print((time.time() - t) / 10, "s per replicate")
# 0.5084290027618408 s per replicate

from statsmodels.nonparametric.smoothers_lowess import lowess
x = np.asarray(s.X_e["Cost per day of work lost"], dtype=float)
u = compute_utilities(s.Y_e)
for delta in [0.0, 0.005 * np.ptp(x), 0.01 * np.ptp(x)]:
    t = time.time()
    lowess(u[:, 0], x, frac=0.3, it=0, delta=delta, return_sorted=False)
    print(f"delta={delta:.3g}: {time.time() - t:.2f}s")