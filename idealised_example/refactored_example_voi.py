# Refactored code for applying VoI to the heat stress example
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

os.chdir('./idealised_example')
from python_funcs import *

# ----- Constants and global set-up ------

# Data folder
DATA_DIR = "/home/aw23877/Documents/bda_sensitivity_paper/bda_risk_dec_sensitivity/data/"

# Number of decisions
nd = 3

# Define X
# Define risk inputs
calibration_opts = ["UKCP_raw", "UKCP_BC", "ChangeFactor"]
warming_opts = ["2deg", "4deg"]
ssp_opts = ["1", "2", "5"]
vuln1_opts = ["53.78", "54.5", "55.79"]
vuln2_opts = ["-4.597", "-4.1", "-3.804"]

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

# Ranges for annual cost per person for each decision
AC_lows = [0, 150, 500]
AC_highs = [0, 350, 700]
# Ranges for efficacies for each decision
E_lows = [0, 0.3, 0.7]
E_highs = [0, 0.5, 0.9]

# Get land indices
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

# ---- Sanity checking ----
# Let's look at the optimal decision in every location under uncertainty
# We will loop over each location and sample from the risk and decision input distributions to get a distribution of Y_e for each decision, then find the optimal decision under uncertainty (the one with lowest expected Y_e) and see how it varies across locations. This is just to check that we are getting different optimal decisions in different locations, and that the optimal decision is not always the same across all locations (which would make VoI less interesting).
# Plot the 1650 - 1660 to see where we are
# plot_index(range(1650, 1660), lat, lon)
# plt.show()

# opt_dec_locs = []
# for idx in range(1711): # just looking at the first 10 locations for now
#     N_samples = 100
#     Y_e_samples = np.empty((nd, N_samples))
#     risk_samples = [
#         np.random.choice(calibration_opts, size=N_samples, replace=True),
#         np.random.choice(warming_opts, size=N_samples, replace=True),
#         np.random.choice(ssp_opts, size=N_samples, replace=True),
#         np.random.choice(vuln1_opts, size=N_samples, replace=True),
#         np.random.choice(vuln2_opts, size=N_samples, replace=True)
#     ]
#     DC_samples = np.random.uniform(low=100, high=300, size=N_samples)
    
#     for d in range(nd):
#         AC_samps = np.random.uniform(low=AC_lows[d], high=AC_highs[d], size=N_samples)
#         E_samps = np.random.uniform(low=E_lows[d], high=E_highs[d], size=N_samples)
        
#         for i in range(N_samples):
#             Y_e_samples[d,i] = calc_Ye(
#                 index = idx,
#                 ind = ind,
#                 input_data_path = DATA_DIR,
#                 risk_inputs = [risk_samples[j][i] for j in range(5)],
#                 decision_inputs = [DC_samples[i], AC_samps[i], E_samps[i]]
#             )
#     expected_losses = np.mean(Y_e_samples, axis=1)
#     optimal_decision_uncertain = np.argmin(expected_losses)
#     opt_dec_locs.append((idx, optimal_decision_uncertain + 1))
#     print(f"Location index: {idx}, Optimal decision under uncertainty: d{optimal_decision_uncertain + 1}")

# # Plot the optimal decision under uncertainty on the map for all locations
# fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
# # Plot decision 1 in blue, decision 2 in orange, decision 3 in green
# for d in range(nd):
#     dec_indices = [idx for idx, opt_dec in opt_dec_locs if opt_dec == d+1]
#     ax.scatter(lon[dec_indices], lat[dec_indices], s=12, label=f'd{d+1}')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Optimal Decision under Uncertainty across Locations')
# ax.legend()
# plt.show()

# opt_dec_locs[1460]
# plot_index(1460, lat, lon)
# plt.show()

# ---- Function for VoI analysis ----
def calculate_evppi(parameter_samples, losses_matrix, optimal_decision_uncertain, n_estimators=50):
    """
    Calculates EVPPI using a Random Forest Regressor.
    parameter_samples: 1D array (N,) of the parameter we are 'learning'
    losses_matrix: 2D array (N, 3) of the Y_e values for each decision
    """
    N, n_decisions = losses_matrix.shape
    # To store E[Loss | Parameter] for each decision
    predicted_expected_losses = np.zeros((N, n_decisions))

    # Check if you can convert the input variable to numeric, and if not, encode it
    try:
        X = parameter_samples.astype(float).reshape(-1, 1)
    except ValueError:
        enc = OrdinalEncoder()
        X = enc.fit_transform(parameter_samples.reshape(-1, 1))
    
    # 1. Fit regression for each decision
    for d in range(n_decisions):
        # Shallow Random Forest to prevent overfitting
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=4, random_state=42)
        model.fit(X, losses_matrix[:, d])
        predicted_expected_losses[:, d] = model.predict(X)
    
    # 2. Determine the optimal decision for EACH sample given perfect info
    # We take the index of the minimum predicted loss
    optimal_decisions_perfect_info = np.argmin(predicted_expected_losses, axis=1)
    utilities_perfect_info = -predicted_expected_losses[np.arange(N), optimal_decisions_perfect_info]
    # Calculate mean utility for the EVPPI check
    expected_utility_perfect_info = np.mean(utilities_perfect_info)

    # 3. Calculate Probability of Decision Change
    # Count how many times the info changed the optimal decision
    num_changes = np.sum(optimal_decisions_perfect_info != optimal_decision_uncertain)
    prob_change = num_changes / N

    # 4. Calculate EVPPI
    avg_losses = np.mean(losses_matrix, axis=0)
    best_avg_loss = np.min(avg_losses)
    expected_loss_perfect_info = np.mean(np.min(predicted_expected_losses, axis=1))
    evppi = best_avg_loss - expected_loss_perfect_info
    
    return evppi, expected_utility_perfect_info, prob_change, utilities_perfect_info

# ------ Function to run the VoI analysis for a single location ------
def run_location_analysis(loc_name, loc_ind, base_N = 1000):
    print(f"Running analysis for {loc_name} (Index: {loc_ind})...")

    # Get EAI and exposure samples for this location across all risk input combinations
    EAI_Exp_samples = get_EAI_Exp_bundle(
        index = loc_ind,
        ind = ind,
        input_data_path = DATA_DIR,
        calibration_opts = calibration_opts,
        warming_level_opts = warming_opts,
        ssp_opts = ssp_opts,
        vuln_param_1_opts = vuln1_opts,
        vuln_param_2_opts = vuln2_opts
    )

    # 1. Set-up the sampling
    Y_e_samples = np.empty((nd, base_N))
    AC_samples = np.empty((nd, base_N))
    E_samples = np.empty((nd, base_N))
    expected_losses = np.empty(nd)
    # Sample the risk inputs
    risk_samples = [
        np.random.choice(calibration_opts, size=base_N, replace=True),
        np.random.choice(warming_opts, size=base_N, replace=True),
        np.random.choice(ssp_opts, size=base_N, replace=True),
        np.random.choice(vuln1_opts, size=base_N, replace=True),
        np.random.choice(vuln2_opts, size=base_N, replace=True)
    ]
    # Sample the cost per day of work
    DC_samples = np.random.uniform(low=100, high=300, size=base_N)

    # 2. Loop over the decisions and calculate base Y_e for each decision
    for d in range(nd):
        print(f'Calculating base losses for decision {d+1}...')
        # Sample the annual cost per person and efficacy for this decision
        AC_samps = np.random.uniform(low=AC_lows[d], high=AC_highs[d], size=base_N)
        E_samps = np.random.uniform(low=E_lows[d], high=E_highs[d], size=base_N)
        AC_samples[d] = AC_samps
        E_samples[d] = E_samps
        
        for i in range(base_N):
            # Get the EAI and exposure samples for this combination of risk inputs
            key = (risk_samples[0][i], risk_samples[1][i], risk_samples[2][i], risk_samples[3][i], risk_samples[4][i])
            EAI_Exp = EAI_Exp_samples[key]
            Y_e_samples[d,i] = calc_Ye_jit(EAI_Exp, [DC_samples[i], AC_samps[i], E_samps[i]])
            # Y_e_samples[d,i] = calc_Ye(
            #     index = loc_ind,
            #     ind = ind,
            #     input_data_path = DATA_DIR,
            #     risk_inputs = [risk_samples[j][i] for j in range(5)],
            #     decision_inputs = [DC_samples[i], AC_samps[i], E_samps[i]]
            # )

        # Calculate the epistemic loss marginalizing epistemic uncertainty
        expected_losses[d] = np.mean(Y_e_samples[d, :])

    expected_utilities_uncertain = -expected_losses
    std_utilities_uncertain = np.std(-Y_e_samples, axis=1)
    optimal_decision_uncertain = np.argmax(expected_utilities_uncertain)

    # 3. Calculate VoI for all parameters
    print("Calculating EVPPI using random forest regression...")
    losses_matrix = Y_e_samples.T # shape (base_N, nd)
    input_samples = {
        'calibration': risk_samples[0],
        'warming': risk_samples[1],
        'ssp': risk_samples[2],
        'vuln1': risk_samples[3],
        'vuln2': risk_samples[4],
        'DC': DC_samples,
        'AC_d2': AC_samples[1],
        'E_d2': E_samples[1],
        'AC_d3': AC_samples[2],
        'E_d3': E_samples[2]
    }
    # For all inputs, calculate VoI
    voi_metrics = {}
    for input_name, samples in input_samples.items():
        voi, expected_utility_perfect_info, prob_change, utilities_with_perfect_info = calculate_evppi(samples, losses_matrix, optimal_decision_uncertain)
        voi_metrics[input_name] = {
            'voi': voi,
            'expected_utility_perfect_info': expected_utility_perfect_info,
            'prob_change': prob_change,
            'utilities_with_perfect_info': utilities_with_perfect_info
        }
    # Save results
    results_dict = {
        'location_name': loc_name,
        'location_index': loc_ind,
        'expected_losses': expected_losses,
        'Y_e_samples': Y_e_samples,
        'expected_utilities_uncertain': expected_utilities_uncertain,
        'std_utilities_uncertain': std_utilities_uncertain,
        'optimal_decision_uncertain': optimal_decision_uncertain,
        'voi_metrics': voi_metrics,
        'inputs': {
            'risk_samples': risk_samples,
            'DC_samples': DC_samples,
            'AC_samples': AC_samples,
            'E_samples': E_samples
        }
    }

    # Save results to a file
    output_path = f"./results/voi_results_{loc_name.replace(' ', '_')}_{loc_ind}.npy"
    np.save(output_path, results_dict)
    print(f"Analysis for {loc_name} completed. Results saved to {output_path}.")

    return results_dict

# Function to generate and save plots and summary statistics for a location given the results dictionary
def generate_location_summary_and_plots(loc_results):
    loc_name = loc_results['location_name']
    print(f"Summary and plots for {loc_name}:")

    # Expected losses for each decision:
    print(f"Expected losses for {loc_name}:")
    for d in range(nd):
        print(f"Decision {d+1}: {loc_results['expected_losses'][d] / 1e6:.2f} million")
    
    # Plot histogram of expected losses for each decision
    plt.figure(figsize=(12, 6))
    for d in range(nd):
        plt.hist(loc_results['Y_e_samples'][d, :], bins=30, alpha=0.5, label=f'Decision {d+1}')
        plt.axvline(loc_results['expected_losses'][d], color=['blue', 'orange', 'green'][d], linestyle='--', label=f'Expected Loss d{d+1}' if d == 0 else None)
    plt.xlabel('Expected Loss (Y_e)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Expected Losses for {loc_name}')
    plt.legend()
    plt.savefig(f"./figures/voi_histogram_{loc_name.replace(' ', '_')}_{loc_results['location_index']}.png")
    plt.show()

    # Plot each input variable vs. Y_e(a) for the 3 decisions
    # Just risk parameters to start
    fig, axs = plt.subplots(3,5, figsize=(15, 10))
    # Get overall minimum and maximum Y_e to set y-limits
    overall_min = np.min(loc_results['Y_e_samples'])
    overall_max = np.max(loc_results['Y_e_samples'])
    for d in range(3): # a row per decision
        # Add row label on the left side
        axs[d,0].text(-0.3, 0.5, f'd={d+1}', transform=axs[d,0].transAxes, 
                      fontsize=12, fontweight='bold', va='center', ha='center', rotation=90)
        
        for i in range(5): # a column per risk input
            x = loc_results['inputs']['risk_samples'][i]
            y = loc_results['Y_e_samples'][d, :]
            x_vals = np.unique(x)
            grouped_y = [y[x == xv] for xv in x_vals]
            axs[d,i].boxplot(grouped_y)
            axs[d, i].set_xticks(range(1, len(x_vals) + 1))
            axs[d, i].set_xticklabels(x_vals)
            axs[d, i].set_xlabel(X_e_labels[i])
            axs[d,i].set_ylabel("Y_e")
            # Set y-limits to the same for all 3 decisions for easier comparison
            axs[d,i].set_ylim(overall_min - 0.01*overall_min, overall_max + 0.01*overall_max)
    plt.tight_layout()
    plt.savefig(f"./figures/voi_risk_inputs_{loc_name.replace(' ', '_')}_{loc_results['location_index']}.png")
    plt.show()

    # Now decision inputs:
    decision_samples = [
        loc_results['inputs']['DC_samples'],
        loc_results['inputs']['AC_samples'][1],
        loc_results['inputs']['AC_samples'][2],
        loc_results['inputs']['E_samples'][1],
        loc_results['inputs']['E_samples'][2],
    ]
    fig, axs = plt.subplots(3,5, figsize=(15, 10))
    for d in range(3): # a row per decision
        # Add row label on the left side
        axs[d,0].text(-0.3, 0.5, f'd={d+1}', transform=axs[d,0].transAxes, 
                      fontsize=12, fontweight='bold', va='center', ha='center', rotation=90)
        
        for i in range(5):
            axs[d,i].scatter(decision_samples[i].astype(float), loc_results['Y_e_samples'][d, :])
            axs[d, i].set_xlabel(X_e_labels[i + 5])
            axs[d,i].set_ylabel("Y_e")
            axs[d,i].set_ylim(overall_min - 0.01*overall_min, overall_max + 0.01*overall_max)
    plt.tight_layout()
    plt.savefig(f"./figures/voi_decision_inputs_{loc_name.replace(' ', '_')}_{loc_results['location_index']}.png")
    plt.show()

    # Expected utility of the optimal decision under uncertainty:
    # print(f"Expected utility of optimal decision under uncertainty for {loc_name}:{loc_results['expected_utilities_uncertain'][loc_results['optimal_decision_uncertain']]:.2f} ± {loc_results['std_utilities_uncertain'][loc_results['optimal_decision_uncertain']]:.2f}")
    print(f"Expected utility of optimal decision under uncertainty for {loc_name}: {loc_results['expected_utilities_uncertain'][loc_results['optimal_decision_uncertain']]:.2f}")
    
    # For each input:
    # Expected utility with perfect information about that input
    # VoI
    # Probability of decision change with perfect information about that input
    for input_name, metrics in loc_results['voi_metrics'].items():
        print(f"\nInput: {input_name}")
        print(f"Expected utility with perfect information about {input_name} for {loc_name}: {metrics['expected_utility_perfect_info']:.2f}")
        print(f"Value of perfect information for {input_name} in {loc_name}: {metrics['voi'] / 1e6:.2f} million")
        print(f"Probability of decision change with perfect information about {input_name} for {loc_name}: {metrics['prob_change']:.2%}")
    
# Let us test this out on London:
lon_name = "London"
lon_ind = 241
timer_start = time.time()
lon_results = run_location_analysis(lon_name, lon_ind, 10000)
timer_end = time.time()
print(f"Time taken for VoI analysis of {lon_name}: {(timer_end - timer_start) / 60:.2f} minutes")
# 0.14 minutes for 10000 samples
# Read in London results
lon_results = np.load(f"./results/voi_results_{lon_name.replace(' ', '_')}_{lon_ind}.npy", allow_pickle=True).item()
generate_location_summary_and_plots(lon_results)

# Get percentage breakdown of the 3 decisions chosen under uncertainty in London
# Y_e_samples has shape (nd, base_N)    
decision_counts = np.bincount(lon_results['Y_e_samples'].argmin(axis=0))
decision_counts / len(lon_results['Y_e_samples'][0, :]) * 100

# Now let's try the Lake District
# ld_name = "Lake District"
# ld_ind = 1058
# timer_start = time.time()
# ld_results = run_location_analysis(ld_name, ld_ind, 10000)
# timer_end = time.time()
# print(f"Time taken for VoI analysis of {ld_name}: {(timer_end - timer_start) / 60:.2f} minutes")
# Read in Lake District results
ld_results = np.load(f"./results/voi_results_{ld_name.replace(' ', '_')}_{ld_ind}.npy", allow_pickle=True).item()
generate_location_summary_and_plots(ld_results)

# Get percentage breakdown of the 3 decisions chosen under uncertainty in the Lake District
decision_counts_ld = np.bincount(ld_results['Y_e_samples'].argmin(axis=0))
decision_counts_ld / len(ld_results['Y_e_samples'][0, :])

# Location in Scotland where d2 was optimal under uncertainty
# scot_name = "Scotland"
# scot_ind = 1460
# timer_start = time.time()
# scot_results = run_location_analysis(scot_name, scot_ind, 10000)
# timer_end = time.time()
# print(f"Time taken for VoI analysis of {scot_name}: {(timer_end - timer_start) / 60:.2f} minutes") # 24.98 minutes for 10000, 100
# Read in Scotland results
scot_results = np.load(f"./results/voi_results_{scot_name.replace(' ', '_')}_{scot_ind}.npy", allow_pickle=True).item()
generate_location_summary_and_plots(scot_results)
