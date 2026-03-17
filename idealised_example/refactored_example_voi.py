# Refactored code for applying VoI to the heat stress example
import matplotlib.pyplot as plt
import os
import numpy as np

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

# ---- Function for VoI analysis ----
def calculate_expected_loss_given_DC_and_decision(DC_val, decision_idx, n_samples):
    Y_e_samps_inner = np.empty(n_samples)
    inner_risk_samples = [
        np.random.choice(calibration_opts, size=n_samples, replace=True),
        np.random.choice(warming_opts, size=n_samples, replace=True),
        np.random.choice(ssp_opts, size=n_samples, replace=True),
        np.random.choice(vuln1_opts, size=n_samples, replace=True),
        np.random.choice(vuln2_opts, size=n_samples, replace=True)
    ]
    inner_AC_samps = np.random.uniform(low=AC_lows[decision_idx], high=AC_highs[decision_idx], size=n_samples)
    inner_E_samps = np.random.uniform(low=E_lows[decision_idx], high=E_highs[decision_idx], size=n_samples)
    
    for i in range(n_samples):
        Y_e_samps_inner[i] = calc_Ye(
                    index=loc_ind,
                    ind=ind,
                    input_data_path=DATA_DIR,
                    risk_inputs=[r[i] for r in inner_risk_samples],
                    decision_inputs=[DC_val, inner_AC_samps[i], inner_E_samps[i]]
        )
    return np.mean(Y_e_samps_inner)

# ------ Function to run the VoI analysis for a single location ------
def run_location_analysis(loc_name, loc_ind, base_N = 1000, n_voi_samples = 100):
    print(f"Running analysis for {loc_name} (Index: {loc_ind})...")

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
            Y_e_samples[d,i] = calc_Ye(
                index = loc_ind,
                ind = ind,
                input_data_path = DATA_DIR,
                risk_inputs = [risk_samples[j][i] for j in range(5)],
                decision_inputs = [DC_samples[i], AC_samps[i], E_samps[i]]
            )

        # Calculate the epistemic loss marginalizing epistemic uncertainty
        expected_losses[d] = np.mean(Y_e_samples[d, :])

    expected_utilities_uncertain = -expected_losses
    optimal_decision_uncertain = np.argmax(expected_utilities_uncertain)

    # 3. Calculate VoI for DC
    print("Calculating VoI for DC...")
    DC_voi_samples = DC_samples[:n_voi_samples]
    utilities_with_perfect_info = []
    decision_changes = 0

    for idx, dc_val in enumerate(DC_voi_samples):
        if idx > 0 and idx % 10 == 0: 
            print(f"VoI progress: {idx}/{n_voi_samples}")
            
        utilities_for_this_dc = []
        for decision_idx in range(nd):
            exp_loss = calculate_expected_loss_given_DC_and_decision(dc_val, decision_idx, n_voi_samples)
            utilities_for_this_dc.append(-exp_loss)
        
        optimal_utility = max(utilities_for_this_dc)
        optimal_decision_for_dc = np.argmax(utilities_for_this_dc)
        
        if optimal_decision_for_dc != optimal_decision_uncertain:
            decision_changes += 1
            
        utilities_with_perfect_info.append(optimal_utility)

    expected_utility_perfect_info = np.mean(utilities_with_perfect_info)
    DC_DC = decision_changes / n_voi_samples
    value_of_perfect_information_DC = expected_utility_perfect_info - expected_utilities_uncertain[optimal_decision_uncertain]
    
    # Save results
    results_dict = {
        'location_name': loc_name,
        'location_index': loc_ind,
        'expected_losses': expected_losses,
        'Y_e_samples': Y_e_samples,
        'expected_utilities_uncertain': expected_utilities_uncertain,
        'optimal_decision_uncertain': optimal_decision_uncertain,
        'voi_metrics': {
            'utilities_with_perfect_info': utilities_with_perfect_info,
            'expected_utility_perfect_info': expected_utility_perfect_info,
            'probability_of_decision_change': DC_DC,
            'value_of_perfect_information_DC': value_of_perfect_information_DC
        },
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

# Let us test this out on London:
loc_name = "London"
loc_ind = 241

loc_results = run_location_analysis(loc_name, loc_ind)

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
    print(f"Expected utility of optimal decision under uncertainty for {loc_name}: {loc_results['expected_utilities_uncertain'][loc_results['optimal_decision_uncertain']]:.2f}")
    # Expected utility with perfect information about DC:
    print(f"Expected utility with perfect information about DC for {loc_name}: {loc_results['voi_metrics']['expected_utility_perfect_info']:.2f}")
    # Show the value of information for DC:
    print(f"Value of perfect information for DC in {loc_name}: {loc_results['voi_metrics']['value_of_perfect_information_DC'] / 1e6:.2f} million")

    # Plot distribution of utilities with perfect information about DC
    plt.figure(figsize=(10, 6))
    plt.hist(loc_results['voi_metrics']['utilities_with_perfect_info'], bins=30, alpha=0.7, label='Utilities with Perfect Info about DC')
    plt.axvline(loc_results['expected_utilities_uncertain'][loc_results['optimal_decision_uncertain']], color='red', linestyle='--', label='Expected Utility under Uncertainty')
    plt.axvline(loc_results['voi_metrics']['expected_utility_perfect_info'], color='green', linestyle='--', label='Expected Utility with Perfect Info')
    plt.xlabel('Utility')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Utilities with Perfect Information about DC for {loc_name}')
    plt.legend()
    plt.savefig(f"./figures/voi_utilities_DC_{loc_name.replace(' ', '_')}_{loc_results['location_index']}.png")
    plt.show() 

    # Show the probability of decision change with perfect information about DC:
    print(f"Probability of decision change with perfect information about DC for {loc_name}: {loc_results['voi_metrics']['probability_of_decision_change']:.2%}")

# Read in London results
lon_results = np.load(f"./results/voi_results_{loc_name.replace(' ', '_')}_{loc_ind}.npy", allow_pickle=True).item()
generate_location_summary_and_plots(lon_results)

