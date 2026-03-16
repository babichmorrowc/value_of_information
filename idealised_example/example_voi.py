# Example VoI in London

import matplotlib.pyplot as plt
import os
import numpy as np
import random

os.chdir('./idealised_example')
from python_funcs import *

# Data folder
DATA_DIR = "/home/aw23877/Documents/bda_sensitivity_paper/bda_risk_dec_sensitivity/data/"

# Select locations of interest
lon_ind = 241 # London
ld_ind = 1058 # Lake District
scot_ind = 1445 # location in Scotland very sensitive to SSP


# Number of decisions
nd = 3

# Define X
# Define risk inputs
calibration_opts = ["UKCP_raw", "UKCP_BC", "ChangeFactor"]
warming_opts = ["2deg", "4deg"]
ssp_opts = ["1", "2", "5"]
vuln1_opts = ["53.78", "54.5", "55.79"]
vuln2_opts = ["-4.597", "-4.1", "-3.804"]

# Ranges for annual cost per person for each decision
AC_lows = [0, 150, 500]
AC_highs = [0, 350, 700]
# Ranges for efficacies for each decision
E_lows = [0, 0.3, 0.7]
E_highs = [0, 0.5, 0.9]

# X labels
# Not worrying about non-financial yet
X_labels = ['Calibration method',
            'Warming level',
            'SSP',
            'Vulnerability parameter 1',
            'Vulnerability parameter 2',
            'Theta', # including risk
            'Cost per day of work lost',
            'Annual cost per person of d2',
            'Effectiveness of d2',
            'Annual cost per person of d3',
            'Effectiveness of d3']

X_e_labels = ['Calibration method',
            'Warming level',
            'SSP',
            'Vulnerability parameter 1',
            'Vulnerability parameter 2',
            'Cost per day of work lost',
            'Annual cost per person of d2',
            'Effectiveness of d2',
            'Annual cost per person of d3',
            'Effectiveness of d3']

# Get land indices
# I think I only have to do this once
Exp_array = get_Exp(input_data_path = DATA_DIR,
                    ssp = ssp_opts[0],
                    ssp_year = 2041)
ind, lat, lon = get_ind_lat_lon(Exp_array,
                                DATA_DIR,
                                data_source = calibration_opts[0],
                                warming_level = warming_opts[0],
                                ssp = ssp_opts[0],
                                vp1 = vuln1_opts[0],
                                vp2 = vuln2_opts[0])

# Calculate Y_e(d)
N = 1000
lon_expected_losses = np.empty(nd)
lon_Y_e_samples = np.empty((nd, N))  # store per-sample Y_e for plotting later
AC_samples = np.empty((nd, N))
E_samples = np.empty((nd, N))
# Sample the risk inputs
risk_samples = [
    np.random.choice(calibration_opts, size=N, replace=True),
    np.random.choice(warming_opts, size=N, replace=True),
    np.random.choice(ssp_opts, size=N, replace=True),
    np.random.choice(vuln1_opts, size=N, replace=True),
    np.random.choice(vuln2_opts, size=N, replace=True)
]
# Sample the cost per day of work
DC_samples = np.random.uniform(low=100, high=300, size=N)
# Loop over the decisions
for d in range(nd):
    print('Calculating for decision:', d+1)
    # Sample cost per person per year for decision d
    AC_samps = np.random.uniform(low = AC_lows[d], high = AC_highs[d], size=N)
    AC_samples[d] = AC_samps
    # Sample efficacy for decision d
    E_samps = np.random.uniform(low = E_lows[d], high=E_highs[d], size=N)
    E_samples[d] = E_samps

    # Calculate the Y_e values
    for i in range(N):
        if(i%100 == 0): print(i)
        lon_Y_e = calc_Ye(
                    index=lon_ind,
                    # index=scot_ind,
                    # index=ld_ind,
                    ind=ind,
                    input_data_path=DATA_DIR,
                    risk_inputs=[r[i] for r in risk_samples],
                    decision_inputs=[DC_samples[i],AC_samps[i],E_samps[i]]
        )
        lon_Y_e_samples[d,i] = lon_Y_e
    
    # average over samples -> expected loss marginalizing epistemic uncertainty
    lon_expected_losses[d] = np.mean(lon_Y_e_samples[d,:])

print("Expected losses (10^6 pounds):", lon_expected_losses / 1e6)

# Plot histograms of expected losses
plt.hist(lon_Y_e_samples[0,:], bins=50, alpha=0.5, label='Do nothing')
plt.hist(lon_Y_e_samples[1,:], bins=50, alpha=0.5, label='Modify working hours')
plt.hist(lon_Y_e_samples[2,:], bins=50, alpha=0.5, label='Buy cooling equipment')
plt.axvline(x=lon_expected_losses[0], color='blue', linestyle='dashed')
plt.axvline(x=lon_expected_losses[1], color='orange', linestyle='dashed')
plt.axvline(x=lon_expected_losses[2], color='green', linestyle='dashed')

plt.legend()
# plt.savefig('figures/Y_e_dist.png')
plt.show()

# # The spread for decision 1 is very wide
# # Let's look at what inputs correspond to the minimum and maximum Y_e for that decision
# # Minimum:
# min_Y_e_idx_d1 = np.argmin(Y_e_samples[0,:])
# risk_inputs_min_Y_e_d1 = [r[min_Y_e_idx_d1] for r in risk_samples]
# decision_inputs_min_Y_e_d1 = [DC_samples[min_Y_e_idx_d1], AC_samples[0, min_Y_e_idx_d1], E_samples[0, min_Y_e_idx_d1]]
# print("Risk inputs for minimum Y_e in decision 1:", risk_inputs_min_Y_e_d1)
# print("Decision inputs for minimum Y_e in decision 1:", decision_inputs_min_Y_e_d1)
# # Maximum:
# max_Y_e_idx_d1 = np.argmax(Y_e_samples[0,:])
# risk_inputs_max_Y_e_d1 = [r[max_Y_e_idx_d1] for r in risk_samples]
# decision_inputs_max_Y_e_d1 = [DC_samples[max_Y_e_idx_d1], AC_samples[0, max_Y_e_idx_d1], E_samples[0, max_Y_e_idx_d1]]
# print("Risk inputs for maximum Y_e in decision 1:", risk_inputs_max_Y_e_d1)
# print("Decision inputs for maximum Y_e in decision 1:", decision_inputs_max_Y_e_d1)

# # Let's plot just the spread for decision 1 and color by warming level
# # As a histogram
# plt.figure(figsize=(10, 6))
# for warming_level in warming_opts:
#     indices = np.where(risk_samples[1] == warming_level)[0]
#     plt.hist(Y_e_samples[0, indices], bins=50, alpha=0.5, label=f'Warming level: {warming_level}')
# plt.axvline(x=expected_losses[0], color='black', linestyle='dashed', label='Expected loss (Do nothing)')
# plt.legend()
# plt.xlabel('Expected loss (Y_e)')
# plt.ylabel('Frequency')
# plt.title('Distribution of Expected Losses for Decision 1 (Do nothing) by Warming Level')
# plt.show()

# Plot each input variable vs. Y_e(a) for the 3 decisions
# Just risk parameters to start
fig, axs = plt.subplots(3,5, figsize=(15, 10))
for d in range(3): # a row per decision
    # Add row label on the left side
    axs[d,0].text(-0.3, 0.5, f'd={d+1}', transform=axs[d,0].transAxes, 
                  fontsize=12, fontweight='bold', va='center', ha='center', rotation=90)
    
    for i in range(5):
        x = risk_samples[i]
        y = lon_Y_e_samples[d]
        x_vals = np.unique(risk_samples[i])
        grouped_y = [y[x == xv] for xv in x_vals]

        axs[d,i].boxplot(grouped_y)
        axs[d, i].set_xticks(range(1, len(x_vals) + 1))
        axs[d, i].set_xticklabels(x_vals)
        axs[d, i].set_xlabel(X_e_labels[i])
        axs[d,i].set_ylabel("Y_e")
        # axs[d,i].set_ylim(0,5.1e8)
        # axs[d,i].set_ylim(-1.7e8,1.8e6)
plt.tight_layout()
# plt.savefig('figures/Y_e_vs_risk.png')
plt.show()

# Then decision parameters
decision_samples = [DC_samples, AC_samples[1], AC_samples[2], E_samples[1], E_samples[2]]
fig, axs = plt.subplots(3,5, figsize=(15, 10))
for d in range(3): # a row per decision
    # Add row label on the left side
    axs[d,0].text(-0.3, 0.5, f'd={d+1}', transform=axs[d,0].transAxes, 
                  fontsize=12, fontweight='bold', va='center', ha='center', rotation=90)
    
    for i in range(5):
        axs[d,i].scatter(decision_samples[i].astype(float), lon_Y_e_samples[d])
        axs[d, i].set_xlabel(X_e_labels[i + 5])
        axs[d,i].set_ylabel("Y_e")
        axs[d,i].set_ylim(0,5.1e8)
        # axs[d,i].set_ylim(-1.7e8,1.8e6)
plt.tight_layout()
plt.savefig('figures/Y_e_vs_decision.png')
plt.show()

# Calculate value of information
# for cost per day of work
def calculate_expected_loss_given_DC_and_decision(index, DC_val, decision_idx, n_samples=1000):
    """Calculate expected loss for a given cost per day of work and decision"""
    Y_e_samps = np.empty(n_samples)
    
    # Sample the risk inputs
    risk_samples = [
        np.random.choice(calibration_opts, size=n_samples, replace=True),
        np.random.choice(warming_opts, size=n_samples, replace=True),
        np.random.choice(ssp_opts, size=n_samples, replace=True),
        np.random.choice(vuln1_opts, size=n_samples, replace=True),
        np.random.choice(vuln2_opts, size=n_samples, replace=True)
    ]

    # Sample cost per person per year for decision d
    AC_samps = np.random.uniform(low = AC_lows[decision_idx], high = AC_highs[decision_idx], size=n_samples)
    # Sample efficacy for decision d
    E_samps = np.random.uniform(low = E_lows[decision_idx], high=E_highs[decision_idx], size=n_samples)
    
    # Calculate the Y_e values
    for i in range(n_samples):
        if(i%100 == 0): print(i)
        Y_e = calc_Ye(
                    index=index,
                    ind=ind,
                    input_data_path=DATA_DIR,
                    risk_inputs=[r[i] for r in risk_samples],
                    decision_inputs=[DC_val,AC_samps[i],E_samps[i]]
        )
        Y_e_samps[i] = Y_e
    
    return np.mean(Y_e_samps)

# Step 1: Calculate expected utility under uncertainty (already done above)
lon_expected_utilities_uncertain = - lon_expected_losses
lon_optimal_decision_uncertain = np.argmax(lon_expected_utilities_uncertain) # decision 3 (index 2)
lon_optimal_decision_uncertain

# Step 2: Calculate expected utility with perfect information about DC
# We need to calculate the expected utility for each possible value of DC
# For each DC sample, find the optimal decision and its utility
n_voi_samples = 100 
DC_voi_samples = DC_samples[:n_voi_samples]

lon_utilities_with_perfect_info = []
# Also calculate the probability of decision change (DC_DC)
decision_changes = 0
for idx, dc_val in enumerate(DC_voi_samples):
    if(idx%10 == 0): print(idx)
    # For this specific value, calculate expected utility for each decision
    utilities_for_this_dc = []
    
    for decision_idx in range(3):
        expected_loss = calculate_expected_loss_given_DC_and_decision(index = lon_ind,
                                                                      DC_val = dc_val,
                                                                      decision_idx = decision_idx,
                                                                      n_samples=n_voi_samples)
        utility = -expected_loss
        utilities_for_this_dc.append(utility)
    
    # Choose the optimal decision for this DC value
    optimal_utility = max(utilities_for_this_dc)
    optimal_decision_for_dc = np.argmax(utilities_for_this_dc)
    if optimal_decision_for_dc != lon_optimal_decision_uncertain:
        decision_changes += 1
    lon_utilities_with_perfect_info.append(optimal_utility)

# Expected utility with perfect information about DC
lon_expected_utility_perfect_info = np.mean(lon_utilities_with_perfect_info)

# Probability of decision change when DC is known perfectly
lon_DC_DC = decision_changes / n_voi_samples
print(f"Probability of Decision Change (DC_DC) when DC is known perfectly: {lon_DC_DC:.4f}")

# Value of perfect information
lon_value_of_perfect_information_DC = lon_expected_utility_perfect_info - lon_expected_utilities_uncertain[lon_optimal_decision_uncertain]

print(f"\nValue of Perfect Information Analysis for DC:")
print(f"Expected utility with perfect info about DC: {lon_expected_utility_perfect_info / 1e6:.2f} million GBP")
print(f"Expected utility under uncertainty: {lon_expected_utilities_uncertain[lon_optimal_decision_uncertain] / 1e6:.2f} million GBP")
print(f"Value of Perfect Information for DC: {lon_value_of_perfect_information_DC / 1e6:.2f} million GBP")

# Plot the distribution of utilities with perfect information
plt.figure(figsize=(10, 6))
plt.hist(np.array(lon_utilities_with_perfect_info)/1e6, bins=30, alpha=0.7, density=True, 
         label='Utility with perfect info about DC', color='green')
plt.axvline(lon_expected_utility_perfect_info/1e6, color='green', linestyle='--', 
           label=f'Expected utility with perfect info: {lon_expected_utility_perfect_info/1e6:.2f}')
plt.axvline(lon_expected_utilities_uncertain[lon_optimal_decision_uncertain]/1e6, color='red', linestyle='--', 
           label=f'Expected utility under uncertainty: {lon_expected_utilities_uncertain[lon_optimal_decision_uncertain]/1e6:.2f}')
plt.xlabel('Utility (million GBP)')
plt.ylabel('Density')
plt.title('Distribution of Utilities with Perfect Information about DC')
plt.legend()
plt.grid()
plt.show()