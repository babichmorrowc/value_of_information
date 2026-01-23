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

# Read in Latin hypercube samples
n_samples = 200
# 200 rows (number of samples) x 6 columns (number of parameters)
X_n = np.loadtxt(DATA_DIR + 'lat_hyp_samples_' + str(n_samples) + '.csv', delimiter=',')

# X labels
# Not worrying about non-financial yet
X_labels = ['SSP',
            'Warming level',
            'Calibration method',
            'Vulnerability parameter 1',
            'Vulnerability parameter 2',
            'Theta', # including risk
            'Cost per day of work lost',
            'Annual cost per person of d2',
            'Effectiveness of d2',
            'Annual cost per person of d3',
            'Effectiveness of d3']

X_e_labels = ['SSP',
            'Warming level',
            'Calibration method',
            'Vulnerability parameter 1',
            'Vulnerability parameter 2',
            'Cost per day of work lost',
            'Annual cost per person of d2',
            'Effectiveness of d2',
            'Annual cost per person of d3',
            'Effectiveness of d3']

# Loop over all combinations of risk x decision options
lon_exp_utils = []
X_samples = []
# Loop over risk parameters
for ssp in ssp_opts:
    for warm in warming_opts:
        # Exposure depends on SSP and SSP year (which comes from warming level)
        # Get SSP year to use based on warming level
        if warm == "2deg":
            ssp_year = 2041
        else:
            ssp_year = 2084
        # Get array of exposure in each cell
        Exp_array = get_Exp(input_data_path = DATA_DIR,
                            ssp = ssp,
                            ssp_year = ssp_year)
        for cal in calibration_opts:
            for vuln1 in vuln1_opts:
                for vuln2 in vuln2_opts:
                    # String of risk inputs
                    risk_input_string = 'ssp'+ssp+'_'+warm+'_'+cal+'_v1_'+vuln1+'_v2_'+vuln2
                    print(risk_input_string)

                    # Get array of EAI
                    # 1000 samples in each cell
                    # 1000 samples from f(theta | risk inputs)
                    EAI_array = get_EAI(input_data_path = DATA_DIR,
                                        data_source = cal,
                                        warming_level = warm,
                                        ssp = ssp,
                                        vp1 = vuln1,
                                        vp2 = vuln2)

                    ind, lat, lon = get_ind_lat_lon(Exp_array,
                                                    DATA_DIR,
                                                    data_source = cal,
                                                    warming_level = warm,
                                                    ssp = ssp,
                                                    vp1 = vuln1,
                                                    vp2 = vuln2)
                    # Loop over decision parameters
                    for x in X_n:
                        # Extract decision parameters
                        # Columns of X are:
                        # cost_per_day, d2_1, d2_2, d3_1, d3_2
                        cost_per_day = x[0]
                        dec_attributes = np.array([[0, 0, 5],
                                                   [x[1], x[2], 6],
                                                   [x[3], x[4], 4]])

                        # Find costs and utilities in London
                        # I think that lon_util_fin are the 1000 Y_e(a) values (3 rows x 1000)
                        lon_optdec, lon_exp_util, lon_util_fin, lon_cost = decision_single_cell(
                            ind = ind,
                            index = lon_ind,
                            EAI = EAI_array,
                            Exp = Exp_array,
                            nd = nd,
                            decision_inputs = dec_attributes,
                            cost_per_day = cost_per_day
                        )

                        # Add lon_exp_util
                        lon_exp_utils.append(lon_exp_util)
                        # Record all input variables
                        X_samples.append([
                            ssp,
                            warm,
                            cal,
                            vuln1,
                            vuln2,
                            x[0],
                            x[1],
                            x[2],
                            x[3],
                            x[4]
                        ])

# 32400 rows x 3 columns
lon_exp_util = np.array(lon_exp_utils)
expected_utilities_uncertain = np.mean(lon_exp_util, axis = 0)
#32400 rows x 10 columns
X_samples = np.array(X_samples)

# Plot London utilities for the 3 decisions
# Histograms of cost for the 3 decisions
plt.hist(lon_exp_util[:,0], bins=50, alpha=0.5, label='Do nothing')
plt.hist(lon_exp_util[:,1], bins=50, alpha=0.5, label='Modify working hours')
plt.hist(lon_exp_util[:,2], bins=50, alpha=0.5, label='Buy cooling equipment')
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
        x = X_samples[:,i]
        y = -lon_exp_util[:, d]
        x_vals = np.unique(X_samples[:,i])
        grouped_y = [y[x == xv] for xv in x_vals]

        axs[d,i].boxplot(grouped_y)
        axs[d, i].set_xticks(range(1, len(x_vals) + 1))
        axs[d, i].set_xticklabels(x_vals)
        axs[d, i].set_xlabel(X_e_labels[i])
        axs[d,i].set_ylabel("Y_e")
        axs[d,i].set_ylim(0,5.4e8)
plt.tight_layout()
plt.show()

# Then decision parameters
fig, axs = plt.subplots(3,5, figsize=(15, 10))
for d in range(3): # a row per decision
    # Add row label on the left side
    axs[d,0].text(-0.3, 0.5, f'd={d+1}', transform=axs[d,0].transAxes, 
                  fontsize=12, fontweight='bold', va='center', ha='center', rotation=90)
    
    for i in range(5):
        axs[d,i].scatter(X_samples[:,i + 5].astype(float), -lon_exp_util[:,d])
        axs[d, i].set_xlabel(X_e_labels[i + 5])
        axs[d,i].set_ylabel("Y_e")
        axs[d,i].set_ylim(0,5.4e8)
plt.tight_layout()
plt.show()

# Calculate value of information
# for SSP

# Step 1: Calculate expected utility under uncertainty (already done above)
optimal_decision_uncertain = np.argmax(expected_utilities_uncertain) # decision 3 (index 2)

# Step 2: Calculate expected utility with perfect information about SSP
# We need to calculate the expected utility for each possible value of SSP
# For each SSP sample, find the optimal decision and its utility
N = 500
n = 200
SSP_samples = random.choices(ssp_opts, k=N)
for n in range(500):
    for d in range(nd):
        