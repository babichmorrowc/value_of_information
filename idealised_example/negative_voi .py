# Code to troubleshoot the negative VoI values
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import skew

import config
from location_funcs import plot_index
from sampling import EpistemicSamples
from precompute_samples import load_precomputed
from utility import compute_utilities
from modified_pawn import encode_Xe_numeric, compute_pawn_indices, plot_pawn_bargraph
import run_modified_pawn_all_locs as rpa
import run_voi_all_locs as run_voi
from voi import compute_all_evppi, fit_smoothing_estimator, plot_smoothing_estimator

# ---- Load samples ----
samples = load_precomputed()

# Get lat lons for mapping
latitudes = samples["lat"]
longitudes = samples["lon"]

# ---- Load VoI results ----
voi_results = run_voi.load_voi_results("voi_results.npz")

# ---- Plotting set-up ----
# Set up colors for plotting
cols = ListedColormap(config.DECISION_COLORS)

# Input labels
risk_inputs = config.X_E_LABELS[:5]
decision_inputs = config.X_E_LABELS[5:]

# ---- Functions ----
def evppi_diagnostics(s, label, frac=0.3):
    x = s.X_e[label]
    u = compute_utilities(s.Y_e)                  # (N, n_decisions)
    N, D = u.shape
    smoothed = np.column_stack(
        [fit_smoothing_estimator(x, u[:, d], label, frac=frac) for d in range(D)]
    )
    d_star = smoothed.argmax(axis=1)
    d_opt = int(u.mean(axis=0).argmax())
    diff = u[np.arange(N), d_star] - u[:, d_opt]  # paired, per-sample
    return dict(
        evppi=diff.mean(),
        se=diff.std(ddof=1) / np.sqrt(N),
        d_opt=d_opt,
        mean_u_dopt=u[:, d_opt].mean(),
        sd_u_dopt=u[:, d_opt].std(),
        mean_u_by_decision=u.mean(axis=0),
        prob_change=np.mean(d_star != d_opt),
    )

def z_for(samples, loc, label, frac=0.3):
    s = EpistemicSamples(
        X_e={k: samples["X_e"][k] for k in config.X_E_LABELS},
        Y_e=samples["Y_e_all"][loc], location_index=loc, n_samples=samples["n_samples"],
    )
    d = evppi_diagnostics(s, label, frac=frac)
    return d["evppi"] / d["se"]

def smooth(x, u_d, label, frac=0.3, it=3):
    x = np.asarray(x, dtype=float)
    if label in CAT:
        return fit_smoothing_estimator(x, u_d, label, frac=frac)
    return lowess(u_d, x, frac=frac, it=it, delta=0.01 * np.ptp(x), return_sorted=False)

def diag(s, label, frac=0.3, it=3):
    x = np.asarray(s.X_e[label]); u = compute_utilities(s.Y_e); N, D = u.shape
    sm = np.column_stack([smooth(x, u[:, d], label, frac, it) for d in range(D)])
    d_star = sm.argmax(axis=1); d_opt = int(u.mean(axis=0).argmax())
    diff = u[np.arange(N), d_star] - u[:, d_opt]
    return diff.mean(), diff.mean() / (diff.std(ddof=1) / np.sqrt(N))

# ---- Negative VoI troubleshooting ----

# Try a non-log-scaled version of the VoI map
fig = plt.figure(figsize=(18,8))
for i, input_i in enumerate(voi_results['labels']):
    ax = fig.add_subplot(2, 5, i + 1, projection=ccrs.PlateCarree())
    scatter = ax.scatter(longitudes, latitudes, c=voi_results['evppi'][:, i], s=10, cmap="Blues")
    ax.set_title(f"({chr(97 + i)})")
    ax.coastlines(linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax, shrink = 0.8)
    cbar.set_label(f"$V_{{X_i}}$ for {voi_results['labels'][i]}")
plt.tight_layout()
plt.show()

# Map of negative VoI values
fig = plt.figure(figsize=(18,8))
for i, input_i in enumerate(voi_results['labels']):
    ax = fig.add_subplot(2, 5, i + 1, projection=ccrs.PlateCarree())
    negative_mask = voi_results['evppi'][:, i] < 0
    scatter = ax.scatter(longitudes[negative_mask], latitudes[negative_mask], c=voi_results['evppi'][negative_mask, i], s=10, cmap="Reds")
    ax.set_title(f"({chr(97 + i)})")
    ax.coastlines(linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax, shrink = 0.8)
    cbar.set_label(f"Negative $V_{{X_i}}$ for {voi_results['labels'][i]}")
plt.tight_layout()
plt.show()

# Check all places and inputs for which VoI is negative
negative_voi_indices = np.where(voi_results['evppi'] < 0)
print("Unique inputs with negative VoI:")
print(np.unique(negative_voi_indices[1])) # just in the continous inputs

# Where is the largest negative VoI value?
print("Largest negative VoI value:")
print(np.min(voi_results['evppi']))

evppi = voi_results["evppi"]                      # (n_locations, n_inputs)
loc, inp = np.unravel_index(np.argmin(evppi), evppi.shape)
label = voi_results["labels"][inp]
print(f"Min EVPPI {evppi[loc, inp]:.0f} at location {loc}, input '{label}'")
print("All EVPPIs at that location:", dict(zip(voi_results["labels"], evppi[loc].round(0))))


s = EpistemicSamples(
    X_e={label: samples["X_e"][label] for label in config.X_E_LABELS},
    Y_e=samples["Y_e_all"][loc],
    location_index=loc,
    n_samples=samples['n_samples']
)
for label in ["Annual cost per person of d3", "Annual cost per person of d2"]:
    d = evppi_diagnostics(s, label)
    print(label)
    print(f"  EVPPI {d['evppi']:.0f}, SE {d['se']:.0f}, z {d['evppi']/d['se']:.2f}")
    print(f"  d_opt {d['d_opt']}, prob_change {d['prob_change']:.3f}")
    print("  mean utility by decision:", d["mean_u_by_decision"].round(0))

label = "Annual cost per person of d3"
inp = list(voi_results["labels"]).index(label)
print(evppi_diagnostics(s, label)["evppi"], voi_results["evppi"][loc, inp])



labels = voi_results["labels"]
locs, inps = np.where(voi_results["evppi"] < 0)
zs = np.array([z_for(samples, l, labels[i]) for l, i in zip(locs, inps)])
print(f"{len(zs)} negative pairs; fraction with z < -2: {np.mean(zs < -2):.1%}; min z: {zs.min():.2f}")
# 380 negative pairs; fraction with z < -2: 54.5%; min z: -30.36

CAT = {"Calibration method", "Warming level", "SSP",
       "Vulnerability parameter 1", "Vulnerability parameter 2"}

# the 20 most negative-z pairs
order_zs = np.argsort(zs)[:20]
for k in order_zs:
    l, i = locs[k], inps[k]
    s = EpistemicSamples(X_e={q: samples["X_e"][q] for q in config.X_E_LABELS},
                         Y_e=samples["Y_e_all"][l], location_index=l, n_samples=samples["n_samples"])
    e3, z3 = diag(s, labels[i], it=3)
    e0, z0 = diag(s, labels[i], it=0)
    print(f"loc {l:4d} {labels[i]:<30} it=3: {e3:>10.0f} (z {z3:6.1f})   it=0: {e0:>10.0f} (z {z0:6.1f})")

# Plot the locations that have negative voi
plot_index(np.unique(locs), latitudes, longitudes)
plt.show()

# Plot the locations with the most negative z values
plot_index(locs[order_zs], latitudes, longitudes)
plt.show()

# ---- Visualize for a single location ----
LOC = 1449                          # one of the worst-z locations
LABEL = "Cost per day of work lost"  # a continuous input; it=3 gave the same EVPPI for every input here
FRAC = 0.3
N_GRID = 200

plot_index(LOC, latitudes, longitudes)
plt.show()

# ---- Build samples for this location (X_e shared across locations) ----
samples = load_precomputed()
s = EpistemicSamples(
    X_e={k: samples["X_e"][k] for k in config.X_E_LABELS},
    Y_e=samples["Y_e_all"][LOC],
    location_index=LOC,
    n_samples=samples["n_samples"],
)
x = np.asarray(s.X_e[LABEL], dtype=float)
u = compute_utilities(s.Y_e)  # (N, D)
N, D = u.shape
grid = np.linspace(x.min(), x.max(), N_GRID)

S3 = np.column_stack([smooth(x, u[:, d], LABEL, frac=0.3, it=3) for d in range(D)])
S0 = np.column_stack([smooth(x, u[:, d], LABEL, frac=0.3, it=0) for d in range(D)])

# fitted values are in sample order; reorder by x for plotting
order = np.argsort(x)
grid = x[order]
S3 = S3[order]
S0 = S0[order]

# ---- Numbers to go with the plots ----
print(f"Location {LOC}, input '{LABEL}', N = {N}")
for d in range(D):
    ud = u[:, d]
    print(
        f"{config.DECISION_LABELS[d]:<20} mean {ud.mean():>14,.0f}  median {np.median(ud):>14,.0f}  "
        f"mean-median {ud.mean() - np.median(ud):>12,.0f}  skew {skew(ud):6.2f}  "
        f"1st pct {np.percentile(ud, 1):>14,.0f}  min {ud.min():>14,.0f}"
    )
print("Decision with best MEAN  :", int(u.mean(axis=0).argmax()) + 1) # decision 2
print("Decision with best MEDIAN:", int(np.median(u, axis=0).argmax()) + 1) # decision 1
print("Share of grid points where each decision wins, it=3:", np.bincount(S3.argmax(1), minlength=D) / N_GRID)
print("Share of grid points where each decision wins, it=0:", np.bincount(S0.argmax(1), minlength=D) / N_GRID)

# ---- Figure ----
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 6, figure=fig)
 
# Row 1: histograms (log counts shows the tail)
for d in range(D):
    ax = fig.add_subplot(gs[0, 2 * d : 2 * d + 2])
    ax.hist(u[:, d], bins=100, color=config.DECISION_COLORS[d], alpha=0.8)
    ax.axvline(u[:, d].mean(), color="black", lw=2, label="mean")
    ax.axvline(np.median(u[:, d]), color="black", lw=2, ls="--", label="median")
    ax.set_yscale("log")
    ax.set_xlabel("Utility")
    ax.set_ylabel("Count (log scale)" if d == 0 else "")
    ax.set_title(config.DECISION_LABELS[d])
    if d == 0:
        ax.legend(fontsize=8)
 
# Row 2: scatter + both smoothers per decision
# y-limits are set to central percentiles so the tail doesn't squash the curves
lo, hi = np.percentile(u, [0.5, 99.5])
pad = 0.05 * (hi - lo)
for d in range(D):
    ax = fig.add_subplot(gs[1, 2 * d : 2 * d + 2])
    ax.scatter(x, u[:, d], s=4, alpha=0.15, color=config.DECISION_COLORS[d])
    ax.plot(grid, S3[:, d], color="red", lw=2, label="LOESS it=3 (robust)")
    ax.plot(grid, S0[:, d], color="black", lw=2, label="LOESS it=0 (mean)")
    ax.axhline(u[:, d].mean(), color="grey", lw=1, ls=":", label="sample mean")
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(LABEL)
    ax.set_ylabel("Utility" if d == 0 else "")
    ax.set_title(f"{config.DECISION_LABELS[d]} (y clipped to 0.5-99.5 pct)")
    if d == 0:
        ax.legend(fontsize=8)
 
# Row 3: overlay all decisions, it=3 vs it=0
for j, (S, name) in enumerate([(S3, "it=3 (robust)"), (S0, "it=0 (conditional mean)")]):
    ax = fig.add_subplot(gs[2, 3 * j : 3 * j + 3])
    for d in range(D):
        ax.plot(grid, S[:, d], color=config.DECISION_COLORS[d], lw=2, label=config.DECISION_LABELS[d])
        ax.axhline(u[:, d].mean(), color=config.DECISION_COLORS[d], lw=1, ls=":")
    ax.set_xlabel(LABEL)
    ax.set_ylabel("Smoothed utility S(x, d)" if j == 0 else "")
    ax.set_title(f"All decisions, {name} (dotted = sample mean)")
    if j == 0:
        ax.legend(fontsize=8)
 
fig.suptitle(f"Location {LOC}: utility distribution and smoothing estimators for '{LABEL}'")
fig.tight_layout()
plt.savefig(config.FIGURES_DIR / f"location_{LOC}_{LABEL.replace(' ', '_')}_diagnostics.png")
plt.show()