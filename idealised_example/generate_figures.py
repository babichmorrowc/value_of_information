# Generate figures for Chapter 4
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import numpy as np
from safepython.PAWN_pmf import pawn_plot_pmf

import config
from sampling import EpistemicSamples
from precompute_samples import load_precomputed
from utility import compute_utilities
from modified_pawn import encode_Xe_numeric, compute_pawn_indices, plot_pawn_bargraph
from voi import compute_all_evppi, plot_smoothing_estimator

# ---- Load samples ----
samples = load_precomputed()

# Get lat lons for mapping
latitudes = samples["lat"]
longitudes = samples["lon"]

# ---- Plotting set-up ----
# Set up colors for plotting
cols = ListedColormap(config.DECISION_COLORS)

# Input labels
risk_inputs = config.X_E_LABELS[:5]
decision_inputs = config.X_E_LABELS[5:]

# ---- Map of optimal decision under uncertainty ----
# Using the precomputed samples, plot the optimal decision under uncertainty for each location

# For each location, find the decision with the lowest mean Y_e
mean_Y_e = samples["Y_e_all"].mean(axis=1)  # shape (n_locations, n_decisions)
optimal_decision_indices = mean_Y_e.argmin(axis=1)  # shape (n_locations,)

# Create a map of the optimal decision under uncertainty
# Using latitude and longitude values from samples
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
scatter = ax.scatter(longitudes, latitudes, c=optimal_decision_indices, s=10, cmap=cols)
# ax.set_title("Optimal Decision under Uncertainty")
ax.legend(handles=scatter.legend_elements()[0], labels=config.DECISION_NAMES)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.coastlines(linewidth=0.5)
plt.savefig(config.FIGURES_DIR / "optimal_decision_uncertainty_map.png")
plt.show()

# ---- Map of percentage optimality for each decision ----
# For each location, compute the percentage of the 10,000 samples where each decision is optimal
percentages_array = np.zeros((len(samples["location_indices"]), config.N_DECISIONS))
for loc in range(len(samples["location_indices"])):
    loc_samples = samples["Y_e_all"][loc]  # shape (n_samples, n_decisions)
    optimal_decisions = loc_samples.argmin(axis=1)  # shape (n_samples,)
    counts = np.bincount(optimal_decisions, minlength=config.N_DECISIONS)
    percentages = 100 * counts / samples["n_samples"]
    percentages_array[loc] = percentages

fig = plt.figure(figsize=(18, 9))
for d in range(config.N_DECISIONS):
    ax = fig.add_subplot(2, 3, d + 1, projection=ccrs.PlateCarree())
    sc = ax.scatter(longitudes, latitudes, c=percentages_array[:, d], cmap="Greens", s=10, vmin=0, vmax=100)
    ax.set_title(f"({chr(97 + d)})")
    ax.coastlines(linewidth=0.5)
    cbar = plt.colorbar(sc, ax=ax, shrink = 0.8)
    cbar.set_label(f"Decision {d + 1}: % optimal")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "percentage_optimality_map.png")
plt.show()

# ---- Histograms of Y_e(d) for selected locations ----
# For 3 locations, plot histograms of Y_e(d) for each decision
fig = plt.figure(figsize=(15, 5))
for i, location_index in enumerate(config.LOCATION_INDICES):  # London, Lake District, Scotland
    samples_for_location = {
        "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
        "Y_e": samples['Y_e_all'][location_index],
        "location_index": location_index,
        "n_samples": samples["n_samples"],
    }
    samples_for_location = EpistemicSamples(**samples_for_location)
    ax = fig.add_subplot(1, 3, i + 1)
    for d in range(config.N_DECISIONS):
        ax.hist(samples_for_location.Y_e[:, d], bins=30, color = cols(d), alpha=0.5, label=config.DECISION_LABELS[d])
        ax.axvline(samples_for_location.Y_e[:, d].mean(), linestyle="--", color=cols(d))
    ax.set_xlabel("Y_e(d)")
    ax.set_ylabel("Count")
    ax.set_title(f"({chr(97 + i)})")
    ax.legend()
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "Ye_distribution_histograms.png")
plt.show()

# ---- Modified PAWN: Plot PMFs for three locations x each epistemic input ----
# Use the SAFE helper for each location, then move its 10 PMF axes into the
# single 6x5 composite figure. This avoids re-implementing the PMF logic while
# still producing one combined layout rather than three standalone figures.
# combined_fig = plt.figure(figsize=(18, 12))
# outer_grid = combined_fig.add_gridspec(6, 5)

# for loc_block, location_index in enumerate(config.LOCATION_INDICES):
#     samples_for_location = {
#         "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
#         "Y_e": samples['Y_e_all'][location_index],
#         "location_index": location_index,
#         "n_samples": samples["n_samples"],
#     }
#     X_numeric = encode_Xe_numeric(samples_for_location["X_e"])
#     utilities = compute_utilities(samples_for_location["Y_e"])
#     Y = np.argmax(utilities, axis=1).astype(float)

#     pawn_plot_pmf(
#         X=X_numeric,
#         Y=Y,
#         n=config.PAWN_N_CONDITIONING_INTERVALS,
#         n_col=5,
#         cbar=False,
#         labelinput=config.X_E_SHORT_LABELS,
#         Y_Label="Optimal decision",
#     )
#     loc_fig = plt.gcf()

#     for ax_index, ax in enumerate(list(loc_fig.axes)):
#         row = ax_index // 5
#         col = ax_index % 5
#         slot = outer_grid[loc_block * 2 + row, col]
#         loc_fig.delaxes(ax)
#         combined_fig.add_axes(ax)
#         ax.set_position(slot.get_position(combined_fig))
#         ax.set_title(f"{config.X_E_SHORT_LABELS[ax_index]}\nloc {location_index}")

#     plt.close(loc_fig)

# combined_fig.tight_layout(rect=[0, 0, 1, 0.97])
# plt.show()

# ---- Modified PAWN: PMF figures for each location saved separately ----
location_index = config.LOCATION_INDICES[0]  # London
samples_for_location = {
    "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
    "Y_e": samples['Y_e_all'][location_index],
    "location_index": location_index,
    "n_samples": samples["n_samples"],
}
X_numeric = encode_Xe_numeric(samples_for_location["X_e"])
utilities = compute_utilities(samples_for_location["Y_e"])
Y = np.argmax(utilities, axis=1).astype(float)

pawn_plot_pmf(
    X=X_numeric,
    Y=Y,
    n_col=5,
    n=config.PAWN_N_CONDITIONING_INTERVALS,
    cbar=True,
    labelinput=config.X_E_SHORT_LABELS,
    Y_Label="Optimal decision",
)
# Add 1 to all x-axis tick labels since decisions are 1-indexed
for ax in plt.gcf().axes[:-1]:
    ax.set_xticks(ticks=range(config.N_DECISIONS), labels=range(1, 1 + config.N_DECISIONS))

plt.savefig(config.FIGURES_DIR / "london_pmf.png")
plt.show()

# Lake District:
location_index = config.LOCATION_INDICES[1]  # Lake District
samples_for_location = {
    "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
    "Y_e": samples['Y_e_all'][location_index],
    "location_index": location_index,
    "n_samples": samples["n_samples"],
}
X_numeric = encode_Xe_numeric(samples_for_location["X_e"])
utilities = compute_utilities(samples_for_location["Y_e"])
Y = np.argmax(utilities, axis=1).astype(float)

pawn_plot_pmf(
    X=X_numeric,
    Y=Y,
    n_col=5,
    n=config.PAWN_N_CONDITIONING_INTERVALS,
    cbar=True,
    labelinput=config.X_E_SHORT_LABELS,
    Y_Label="Optimal decision",
)
# Add 1 to all x-axis tick labels since decisions are 1-indexed
for ax in plt.gcf().axes[:-1]:
    ax.set_xticks(ticks=range(config.N_DECISIONS), labels=range(1, 1 + config.N_DECISIONS))

# plt.savefig(config.FIGURES_DIR / "lakedistrict_pmf.png")
plt.show()

# Scotland:
location_index = config.LOCATION_INDICES[2]  # Scotland
samples_for_location = {
    "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
    "Y_e": samples['Y_e_all'][location_index],
    "location_index": location_index,
    "n_samples": samples["n_samples"],
}
X_numeric = encode_Xe_numeric(samples_for_location["X_e"])
utilities = compute_utilities(samples_for_location["Y_e"])
Y = np.argmax(utilities, axis=1).astype(float)

pawn_plot_pmf(
    X=X_numeric,
    Y=Y,
    n_col=5,
    n=config.PAWN_N_CONDITIONING_INTERVALS,
    cbar=True,
    labelinput=config.X_E_SHORT_LABELS,
    Y_Label="Optimal decision",
)
# Add 1 to all x-axis tick labels since decisions are 1-indexed
for ax in plt.gcf().axes[:-1]:
    ax.set_xticks(ticks=range(config.N_DECISIONS), labels=range(1, 1 + config.N_DECISIONS))

# plt.savefig(config.FIGURES_DIR / "scotland_pmf.png")
plt.show()

# ---- Modified PAWN: Barplot of PAWN sensitivity indices for each epistemic input ----
# For 3 locations, compute the modified PAWN indices and plot them
# In a 3 panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, location_index in enumerate(config.LOCATION_INDICES):  # London, Lake District, Scotland
    samples_for_location = {
        "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
        "Y_e": samples['Y_e_all'][location_index],
        "location_index": location_index,
        "n_samples": samples["n_samples"]
    }
    samples_for_location = EpistemicSamples(**samples_for_location)
    pawn_indices = compute_pawn_indices(samples_for_location)
    ax = plot_pawn_bargraph(pawn_indices, location_index=location_index, ax=axes[i])
    ax.set_xticks(ticks = range(len(config.X_E_SHORT_LABELS)),
                  labels = config.X_E_SHORT_LABELS)
    ax.set_ylim(0, 0.3)
    ax.set_title(f"({chr(97 + i)})")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "pawn_bargraphs.png")
plt.show()

# ---- VoI: Loess smoothing plots for Lake District ----
loc_index = config.LOCATION_INDICES[1]

# Pull samples for the Lake District
samples_for_location = {
    "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
    "Y_e": samples['Y_e_all'][loc_index],
    "location_index": loc_index,
    "n_samples": samples["n_samples"],
}
samples_for_location = EpistemicSamples(**samples_for_location)

# First, plot for the risk-related inputs:
# Create a 3x5 figure
fig, axes = plt.subplots(3, 5, figsize=(20, 6))
panel_label = 0

for row_idx in range(3):
    # Determine which decision (0, 1, or 2)
    decision_idx = row_idx % 3
    
    for col_idx, input_label in enumerate(risk_inputs):
        ax = axes[row_idx, col_idx]
        plot_smoothing_estimator(
            samples_for_location,
            input_label,
            location_index=loc_index,
            ax=ax,
            decision_index=decision_idx,
        )
        # Add panel label
        ax.text(-0.3, 1.05, f"({chr(97 + panel_label)})", transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom', ha='right')
        panel_label += 1

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "voi_conditionalexp_lake_district.png")
plt.show()

# Now the decision-related inputs
fig, axes = plt.subplots(3, 5, figsize=(20, 6))
panel_label = 0

for row_idx in range(3):
    # Determine which decision (0, 1, or 2)
    decision_idx = row_idx % 3
    
    for col_idx, input_label in enumerate(decision_inputs):
        ax = axes[row_idx, col_idx]
        plot_smoothing_estimator(
            samples_for_location,
            input_label,
            location_index=loc_index,
            ax=ax,
            decision_index=decision_idx,
        )
        # Add panel label
        ax.text(-0.3, 1.05, f"({chr(97 + panel_label)})", transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom', ha='right')
        panel_label += 1

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "voi_smoothing_lake_district.png")
plt.show()

# ---- VoI: barplot of VoI values in all 3 locations ----
# In a 3 panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, location_index in enumerate(config.LOCATION_INDICES):  # London, Lake District, Scotland
    samples_for_location = {
        "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
        "Y_e": samples['Y_e_all'][location_index],
        "location_index": location_index,
        "n_samples": samples["n_samples"]
    }
    samples_for_location = EpistemicSamples(**samples_for_location)
    voi_results = compute_all_evppi(samples_for_location)
    voi_vals = [r.evppi for r in voi_results]

    ax = axes[i]
    x = np.arange(len(config.X_E_SHORT_LABELS))
    ax.bar(x, voi_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(config.X_E_SHORT_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Value of information")
    ax.set_title(f"({chr(97 + i)})")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "voi_bargraphs.png")
plt.show()

# ---- VoI: barplot of probability of decision change in all 3 locations ----
# In a 3 panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, location_index in enumerate(config.LOCATION_INDICES):  # London, Lake District, Scotland
    samples_for_location = {
        "X_e": {key: samples['X_e'][key] for key in config.X_E_LABELS},
        "Y_e": samples['Y_e_all'][location_index],
        "location_index": location_index,
        "n_samples": samples["n_samples"]
    }
    samples_for_location = EpistemicSamples(**samples_for_location)
    voi_results = compute_all_evppi(samples_for_location)
    dc_vals = [r.prob_change for r in voi_results]

    ax = axes[i]
    x = np.arange(len(config.X_E_SHORT_LABELS))
    ax.bar(x, dc_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(config.X_E_SHORT_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Probability of decision change")
    ax.set_title(f"({chr(97 + i)})")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "dc_bargraphs.png")
plt.show()