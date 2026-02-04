# Example VoI in London

import matplotlib.pyplot as plt
import os
import numpy as np
import random

os.chdir('./idealised_example')
from python_funcs import *

# Data folder
DATA_DIR = "/home/aw23877/Documents/bda_sensitivity_paper/bda_risk_dec_sensitivity/data/"

# Select London data
lon_ind = 241 # London

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
E_lows = [0, 30, 70]
E_highs = [0, 50, 90]

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
expected_losses = np.empty(nd)
Y_e_samples = np.empty((nd, N))  # store per-sample Y_e for plotting later
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
        Y_e = calc_Ye(
                    index=lon_ind,
                    ind=ind,
                    input_data_path=DATA_DIR,
                    risk_inputs=[r[i] for r in risk_samples],
                    decision_inputs=[DC_samples[i],AC_samps[i],E_samps[i]]
        )
        Y_e_samples[d,i] = Y_e
    
    # average over samples -> expected loss marginalizing epistemic uncertainty
    expected_losses[d] = np.mean(Y_e_samples[d,:])

print("Expected losses (10^9 pounds):", expected_losses / 1e9)

# Plot histograms of expected losses
plt.hist(Y_e_samples[0,:], bins=50, alpha=0.5, label='Do nothing')
plt.hist(Y_e_samples[1,:], bins=50, alpha=0.5, label='Modify working hours')
plt.hist(Y_e_samples[2,:], bins=50, alpha=0.5, label='Buy cooling equipment')
plt.legend()
plt.show()

# Plot each input variable vs. Y_e(a) for the 3 decisions
# Just risk parameters to start
fig, axs = plt.subplots(3,5, figsize=(15, 10))
for d in range(3): # a row per decision
    # Add row label on the left side
    axs[d,0].text(-0.3, 0.5, f'd={d+1}', transform=axs[d,0].transAxes, 
                  fontsize=12, fontweight='bold', va='center', ha='center', rotation=90)
    
    for i in range(5):
        x = risk_samples[i]
        y = Y_e_samples[d]
        x_vals = np.unique(risk_samples[i])
        grouped_y = [y[x == xv] for xv in x_vals]

        axs[d,i].boxplot(grouped_y)
        axs[d, i].set_xticks(range(1, len(x_vals) + 1))
        axs[d, i].set_xticklabels(x_vals)
        axs[d, i].set_xlabel(X_e_labels[i])
        axs[d,i].set_ylabel("Y_e")
        axs[d,i].set_ylim(-4.1e10,5.1e8)
plt.tight_layout()
plt.show()

# Then decision parameters
decision_samples = [DC_samples, AC_samples[1], AC_samples[2], E_samples[1], E_samples[2]]
fig, axs = plt.subplots(3,5, figsize=(15, 10))
for d in range(3): # a row per decision
    # Add row label on the left side
    axs[d,0].text(-0.3, 0.5, f'd={d+1}', transform=axs[d,0].transAxes, 
                  fontsize=12, fontweight='bold', va='center', ha='center', rotation=90)
    
    for i in range(5):
        axs[d,i].scatter(decision_samples[i].astype(float), Y_e_samples[d])
        axs[d, i].set_xlabel(X_e_labels[i + 5])
        axs[d,i].set_ylabel("Y_e")
        axs[d,i].set_ylim(-4.1e10,5.1e8)
plt.tight_layout()
plt.show()

# Calculate value of information
# for cost per day of work
def calculate_expected_loss_given_DC_and_decision(DC_val, decision_idx, n_samples=1000):
    """Calculate expected loss for a given cost per day of work and decision"""
    rng_local = np.random.default_rng(42)  # Fixed seed for reproducibility
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
                    index=lon_ind,
                    ind=ind,
                    input_data_path=DATA_DIR,
                    risk_inputs=[r[i] for r in risk_samples],
                    decision_inputs=[DC_val,AC_samps[i],E_samps[i]]
        )
        Y_e_samps[i] = Y_e
    
    return np.mean(Y_e_samps)

# Step 1: Calculate expected utility under uncertainty (already done above)
expected_utilities_uncertain = - expected_losses
optimal_decision_uncertain = np.argmax(expected_utilities_uncertain) # decision 3 (index 2)

# Step 2: Calculate expected utility with perfect information about DC
# We need to calculate the expected utility for each possible value of DC
# For each DC sample, find the optimal decision and its utility
n_voi_samples = 100 
DC_voi_samples = DC_samples[:n_voi_samples]

utilities_with_perfect_info = []
for dc_val in DC_voi_samples:
    # For this specific value, calculate expected utility for each decision
    utilities_for_this_dc = []
    
    for decision_idx in range(3):
        expected_loss = calculate_expected_loss_given_DC_and_decision(dc_val, decision_idx, n_samples=n_voi_samples)
        utility = -expected_loss
        utilities_for_this_dc.append(utility)
    
    # Choose the optimal decision for this DC value
    optimal_utility = max(utilities_for_this_dc)
    optimal_decision_for_dc = np.argmax(utilities_for_this_dc)
    if optimal_decision_for_dc != optimal_decision_uncertain:
        decision_changes += 1
    utilities_with_perfect_info.append(optimal_utility)


        