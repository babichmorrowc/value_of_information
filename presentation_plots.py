import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, gumbel_r
from scipy.special import expi

# Plot three lognormal distributions for R_1 to R_3
# Parameters for the lognormal distributions
mean_Ra_values = np.array([10, 12, 14])
std_Ra_values = np.array([1, 1, 1])

# Parameters for the underlying normal distribution
mu_values = np.log(mean_Ra_values**2 / np.sqrt(std_Ra_values**2 + mean_Ra_values**2))
sigma_values = np.sqrt(np.log(1 + (std_Ra_values**2 / mean_Ra_values**2)))

labels = ['R_1', 'R_2', 'R_3']
R_colors = ['#4779c4', '#3c649f', '#2c456b']
x = np.linspace(0, 18, 1000)
plt.figure(figsize=(10, 6))
for mu, sigma, label, color in zip(mu_values, sigma_values, labels, R_colors):
    s = sigma
    scale = np.exp(mu) # median of the lognormal distribution
    pdf = lognorm.pdf(x, s=s, scale=scale)
    plt.plot(x, pdf, label=label, color=color)
plt.title('Distributions for resistances')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()

# Plot a Gumbel distribution for S
# With M at its mean value
M_mean = 7.5
M_std = 1
beta = 1
x_s = np.linspace(0, 18, 1000)
pdf_s = gumbel_r.pdf(x_s, loc=M_mean, scale=beta)
plt.figure(figsize=(10, 6))
plt.plot(x_s, pdf_s, label='S', color='#c44747')
plt.title('Gumbel Distribution for Load S (M at mean value)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()

# Overlay S distribution on the R distributions
plt.figure(figsize=(10, 6))
for mu, sigma, label, color in zip(mu_values, sigma_values, labels, colors
):
    s = sigma
    scale = np.exp(mu) # median of the lognormal distribution
    pdf = lognorm.pdf(x, s=s, scale=scale)
    plt.plot(x, pdf, label=label, color=color)
plt.plot(x_s, pdf_s, label='S', color='#c44747', linestyle='--')
plt.title('Distributions for Resistances and Load S (M at mean value)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()   

# The below code doesn't work because I am not accounting for the distributions of M, R_a, or C_F
# # Evaluate the expected loss for the different a values
# c_a_values = np.array([13, 15, 17]) * 10**6
# R_a_values = np.array([10, 12, 14])
# C_F = 3 * 10**7
# M = 7.5
# A_values = np.exp(M - R_a_values)

# expected_losses = C_F * ((M - R_a_values)*(1 - np.exp(-1*A_values)) - expi(-1*A_values) + np.exp(-1*A_values)*np.log(A_values) + np.euler_gamma)
# expected_losses / 10**6

# expected_utilities = -1 * expected_losses - c_a_values
# expected_utilities / 10**6
# np.argmin(expected_utilities) # which is the optimal decision

# ChatGPT attempt
# Problem data (from the paper)
ca_vals = np.array([13, 15, 17]) * 1e6        # protection costs (not used for losses)
mean_CF = 3.0e7
sd_CF   = 1.0e7
C_F_note = None                               # CF is stochastic here; we will sample it

# number of Monte Carlo samples
N = 2000000   # use large N for accuracy; reduce if too slow
# N = 500 # trying to match the paper

# helper to convert (mean, sd) of *lognormal* to underlying normal params
# https://en.wikipedia.org/wiki/Log-normal_distribution
def lognormal_params_from_mean_sd(mean, sd): # mu_X, sigma_X
    var = sd**2
    phi = np.sqrt(var/mean**2 + 1.0)
    sigma2 = np.log(phi**2)
    mu = np.log(mean) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    return mu, sigma

# compute normal params for R and CF
mu_R, sigma_R = lognormal_params_from_mean_sd(mean_Ra_values, std_Ra_values)
mu_CF, sigma_CF = lognormal_params_from_mean_sd(mean_CF, sd_CF)

# draw samples
rng = np.random.default_rng(12345)
# Sample M from N(7.5, 1)
M_samps = rng.normal(loc=M_mean, scale=M_std, size=N)

# sample CF once per sample (same CF used for all alternatives in a given sample)
CF_samps = rng.lognormal(mean=mu_CF, sigma=sigma_CF, size=N)

expected_losses = np.empty(3)
Y_e_samples = np.empty((3, N))  # store per-sample Y_e for plotting later
R_samples = np.empty((3,N))

def compute_Ye(M, R, CF):
    A = np.exp(M - R)
    term = ((M - R) * (1 - np.exp(-A)) - expi(-A)
            + np.exp(-A) * np.log(A) + np.euler_gamma)
    return CF * term

for i in range(3):
    R_samps = rng.lognormal(mean=mu_R[i], sigma=sigma_R[i], size=N)
    R_samples[i] = R_samps

    Y_e_samps = compute_Ye(M_samps, R_samps, CF_samps)
    # Save Y_e_samps for plotting
    Y_e_samples[i] = Y_e_samps

    # average over samples -> expected loss marginalizing epistemic uncertainty
    expected_losses[i] = np.mean(Y_e_samps)

print("Expected losses (euro):", expected_losses)
print("Expected losses (10^6 euro):", expected_losses / 1e6)

# Plot histograms of the three expected loss samples
plt.figure(figsize=(10, 6))
bins = np.linspace(0, 20e6, 200)
offsets = [0.2, 0.2, -1]
for i in range(3):
    plt.hist(Y_e_samples[i]/1e6, bins=bins/1e6, density=True, alpha=0.5, label=labels[i], color=R_colors[i])
    # Vertical dotted lines for the means
    plt.axvline(expected_losses[i]/1e6, color=R_colors[i], linestyle='dotted')
    # Label the dotted lines
    plt.text(expected_losses[i]/1e6 + offsets[i], 6, f'{labels[i]}:\n{expected_losses[i]/1e6:.2f}', color=R_colors[i])
plt.title('Histograms of Expected Losses for Different Protection Levels')
plt.xlabel('Expected Loss (10^6 euro)')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# Plot histograms of the three expected utilities
plt.figure(figsize=(10, 6))
bins = np.linspace(-45e6, -10e6, 200)
# colors = ['red', 'green', 'blue']
offsets = [-1.2, 0.1, 0.2]
for i in range(3):
    plt.hist((-1*Y_e_samples[i] - ca_vals[i])/1e6, bins=bins/1e6, density=True, alpha=0.5, label=labels[i], color=R_colors[i])
    # Vertical dotted lines for the means
    plt.axvline((-1*expected_losses[i] - ca_vals[i])/1e6, color=R_colors[i], linestyle='dotted')
    plt.text((-1*expected_losses[i] - ca_vals[i])/1e6 + offsets[i], 1.75, f'{labels[i]}:\n{(-1*expected_losses[i] - ca_vals[i])/1e6:.2f}', color=R_colors[i])
plt.title('Histograms of Expected Utilities for Different Protection Levels')
plt.xlabel('Expected Utility (10^6 euro)')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(M_samps, Y_e_samples[0])
plt.show()

# Replicate Figure 1 from the paper
x_variables = [M_samps, R_samples[0], R_samples[1], R_samples[2], CF_samps]
x_labels = ["m", "r_1", "r_2", "r_3", "c_F"]
x_lims = [[4,10.5], [7,15], [9,17], [11,18], [0,8.5e7]]
fig, axs = plt.subplots(3,5, figsize=(15, 10))
for a in range(3): # a row per decision
    # Add row label on the left side
    axs[a,0].text(-0.3, 0.5, f'a={a+1}', transform=axs[a,0].transAxes, 
                  fontsize=12, fontweight='bold', va='center', ha='center', rotation=90)
    
    for i in range(len(x_variables)):
        axs[a,i].scatter(x_variables[i], Y_e_samples[a])
        axs[a,i].set_xlabel(x_labels[i])
        axs[a,i].set_xlim(x_lims[i])
        axs[a,i].set_ylabel("Y_e")
        axs[a,i].set_ylim(0,5e7)

plt.tight_layout()
plt.show()

# Calculate value of information
# For M

# Step 1: Calculate expected utility under uncertainty (already done above)
expected_utilities_uncertain = -1 * expected_losses - ca_vals
print("Expected utilities under uncertainty:", expected_utilities_uncertain / 1e6)
optimal_decision_uncertain = np.argmax(expected_utilities_uncertain) # decision 2 (index 1)
print(f"Optimal decision under uncertainty: a={optimal_decision_uncertain + 1}")
print(f"Expected utility under uncertainty: {expected_utilities_uncertain[optimal_decision_uncertain] / 1e6:.2f} million euro")

# Step 2: Calculate expected utility with perfect information about M
# We need to calculate the expected utility for each possible value of M
# For each M sample, find the optimal decision and its utility

def calculate_expected_loss_given_M_and_decision(M_val, decision_idx, n_samples=10000):
    """Calculate expected loss for a given M value and decision"""
    rng_local = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Sample R and CF
    R_samps = rng_local.lognormal(mean=mu_R[decision_idx], sigma=sigma_R[decision_idx], size=n_samples)
    CF_samps = rng_local.lognormal(mean=mu_CF, sigma=sigma_CF, size=n_samples)
    
    Y_e_samps = compute_Ye(M_val, R_samps, CF_samps)
    
    return np.mean(Y_e_samps)

# Calculate value of perfect information
# We'll use a subset of M samples for computational efficiency
n_voi_samples = 1000  # Reduce this if computation is too slow
M_voi_samples = M_samps[:n_voi_samples]

utilities_with_perfect_info = []

print("Calculating value of perfect information for M...")
print("This may take a moment...")

# Also calculate the probability of decision change (DC_M)
decision_changes = 0
for m_val in M_voi_samples:
    # For this specific M value, calculate expected utility for each decision
    utilities_for_this_M = []
    
    for decision_idx in range(3):
        expected_loss = calculate_expected_loss_given_M_and_decision(m_val, decision_idx)
        utility = -expected_loss - ca_vals[decision_idx]
        utilities_for_this_M.append(utility)
    
    # Choose the optimal decision for this M value
    optimal_utility = max(utilities_for_this_M)
    optimal_decision_for_M = np.argmax(utilities_for_this_M)
    if optimal_decision_for_M != optimal_decision_uncertain:
        decision_changes += 1
    utilities_with_perfect_info.append(optimal_utility)

# Expected utility with perfect information about M
expected_utility_perfect_info = np.mean(utilities_with_perfect_info)

# Probability of decision change when M is known perfectly
DC_M = decision_changes / n_voi_samples
print(f"Probability of Decision Change (DC_M) when M is known perfectly: {DC_M:.4f}")

# Value of perfect information
value_of_perfect_information_M = expected_utility_perfect_info - expected_utilities_uncertain[optimal_decision_uncertain]

print(f"\nValue of Perfect Information Analysis for M:")
print(f"Expected utility with perfect info about M: {expected_utility_perfect_info / 1e6:.2f} million euro")
print(f"Expected utility under uncertainty: {expected_utilities_uncertain[optimal_decision_uncertain] / 1e6:.2f} million euro")
print(f"Value of Perfect Information for M: {value_of_perfect_information_M / 1e3:.2f} x10^3 euro")

# Plot the distribution of utilities with perfect information
plt.figure(figsize=(10, 6))
plt.hist(np.array(utilities_with_perfect_info)/1e6, bins=30, alpha=0.7, density=True, 
         label='Utility with perfect info about M', color='green')
plt.axvline(expected_utility_perfect_info/1e6, color='green', linestyle='--', 
           label=f'Expected utility with perfect info: {expected_utility_perfect_info/1e6:.2f}')
plt.axvline(expected_utilities_uncertain[optimal_decision_uncertain]/1e6, color='red', linestyle='--', 
           label=f'Expected utility under uncertainty: {expected_utilities_uncertain[optimal_decision_uncertain]/1e6:.2f}')
plt.xlabel('Utility (million euro)')
plt.ylabel('Density')
plt.title('Distribution of Utilities with Perfect Information about M')
plt.legend()
plt.grid()
plt.show()

# Calculate value of information for R_1
# Step 2: Calculate expected utility with perfect information about R_1
# We need to calculate the expected utility for each possible value of R_1
# For each R_1 sample, find the optimal decision and its utility

def calculate_expected_loss_given_R1_and_decision(R1_val, decision_idx, n_samples=10000):
    """Calculate expected loss for a given R1 value and decision
    
    When R1 is known perfectly, we use that value for decision 0 (a=1),
    but still sample from the appropriate R distribution for other decisions.
    """
    rng_local = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Sample M and CF (always uncertain)
    M_samps = rng_local.normal(loc=M_mean, scale=M_std, size=n_samples)
    CF_samps = rng_local.lognormal(mean=mu_CF, sigma=sigma_CF, size=n_samples)
    
    # For R: use known R1 value if decision_idx == 0, otherwise sample from appropriate distribution
    if decision_idx == 0:
        R_samps = np.full(n_samples, R1_val)  # Use known R1 value
    else:
        R_samps = rng_local.lognormal(mean=mu_R[decision_idx], sigma=sigma_R[decision_idx], size=n_samples)
    
    # # Calculate A and expected loss
    # A = np.exp(M_samps - R_samps)  # Fixed: use M_samps instead of M_val
    # bracket = ((M_samps - R_samps) * (1.0 - np.exp(-A)) 
    #            - expi(-A) 
    #            + np.exp(-A) * np.log(A) 
    #            + np.euler_gamma)
    # Y_e_samps = CF_samps * bracket
    Y_e_samps = compute_Ye(M_samps, R_samps, CF_samps)
    
    return np.mean(Y_e_samps)

# Calculate value of perfect information
# We'll use a subset of M samples for computational efficiency
R1_voi_samps = rng.lognormal(mean=mu_R[0], sigma=sigma_R[0], size=n_voi_samples)

utilities_with_perfect_info = []

decision_changes = 0
for r1_val in R1_voi_samps:
    # For this specific R1 value, calculate expected utility for each decision
    utilities_for_this_R1 = []
    
    for decision_idx in range(3):
        expected_loss = calculate_expected_loss_given_R1_and_decision(r1_val, decision_idx)
        utility = -expected_loss - ca_vals[decision_idx]
        utilities_for_this_R1.append(utility)
    
    # Choose the optimal decision for this M value
    optimal_utility = max(utilities_for_this_R1)
    optimal_decision_for_R1 = np.argmax(utilities_for_this_R1)
    if optimal_decision_for_R1 != optimal_decision_uncertain:
        decision_changes += 1
    utilities_with_perfect_info.append(optimal_utility)

# Expected utility with perfect information about R1
expected_utility_perfect_info = np.mean(utilities_with_perfect_info)

# Probability of decision change when R1 is known perfectly
DC_R1 = decision_changes / n_voi_samples
print(f"Probability of Decision Change (DC_R1) when R1 is known perfectly: {DC_R1:.4f}")

# Value of perfect information
value_of_perfect_information_R1 = expected_utility_perfect_info - expected_utilities_uncertain[optimal_decision_uncertain]

print(f"\nValue of Perfect Information Analysis for R1:")
print(f"Expected utility with perfect info about R1: {expected_utility_perfect_info / 1e6:.2f} million euro")
print(f"Expected utility under uncertainty: {expected_utilities_uncertain[optimal_decision_uncertain] / 1e6:.2f} million euro")
print(f"Value of Perfect Information for R1: {value_of_perfect_information_R1 / 1e3:.2f} x10^3 euro")


# Calculating EVPM

# Calculate expected utility with perfect information about X_e (M, R1, R2, R3, CF)

# def calculate_expected_loss_given_Xe_and_decision(M_val, R_val, CF_val, decision_idx):
#     """Calculate expected loss for a given set of X_e values and decision
    
#     When all epistemic variables (M, R, CF) are known perfectly,
#     the loss calculation becomes deterministic - no sampling needed.
#     """
#     # # Calculate A and expected loss deterministically
#     # A = np.exp(M_val - R_val)
#     # bracket = ((M_val - R_val) * (1.0 - np.exp(-A)) 
#     #            - expi(-A) 
#     #            + np.exp(-A) * np.log(A) 
#     #            + np.euler_gamma)
#     # Y_e = CF_val * bracket
    
#     return Y_e

# Calculate EVPM
# This is the value of having perfect information about the epistemic variables

print("\nCalculating Expected Value of Perfect Model (EVPM)...")

# Use a subset of samples for computational efficiency
n_evpm_samples = 500  # Adjust based on computational constraints

utilities_with_perfect_info_all = []

for i in range(n_evpm_samples):
    # Get the values of all epistemic variables for this sample
    M_val = M_samps[i]
    CF_val = CF_samps[i]
    
    # For this specific realization of M and CF, calculate utility for each decision
    utilities_for_this_sample = []
    
    for decision_idx in range(3):
        R_val = R_samples[decision_idx, i]  # The R value for this decision and sample
        
        # Calculate the exact loss (no uncertainty left)
        loss = compute_Ye(M_val, R_val, CF_val)
        utility = -loss - ca_vals[decision_idx]
        utilities_for_this_sample.append(utility)
    
    # Choose the optimal decision for this specific realization
    optimal_utility = max(utilities_for_this_sample)
    utilities_with_perfect_info_all.append(optimal_utility)

# Expected utility with perfect information about all epistemic variables
expected_utility_perfect_info_all = np.mean(utilities_with_perfect_info_all)

# Expected Value of Perfect Model
evpm = expected_utility_perfect_info_all - expected_utilities_uncertain[optimal_decision_uncertain]

print(f"\nExpected Value of Perfect Information (evpm) Analysis:")
print(f"Expected utility with perfect info about all X_e: {expected_utility_perfect_info_all / 1e6:.2f} million euro")
print(f"Expected utility under uncertainty: {expected_utilities_uncertain[optimal_decision_uncertain] / 1e6:.2f} million euro")
print(f"evpm (all epistemic variables): {evpm / 1e6:.2f} million euro")

# Plot comparison of utilities
plt.figure(figsize=(12, 6))
plt.hist(np.array(utilities_with_perfect_info)/1e6, bins=30, alpha=0.6, density=True, 
         label='Perfect info about M only', color='green')
plt.hist(np.array(utilities_with_perfect_info_all)/1e6, bins=30, alpha=0.6, density=True, 
         label='Perfect info about all X_e', color='blue')

plt.axvline(expected_utility_perfect_info/1e6, color='green', linestyle='--', 
           label=f'E[U] with perfect M: {expected_utility_perfect_info/1e6:.2f}')
plt.axvline(expected_utility_perfect_info_all/1e6, color='blue', linestyle='--', 
           label=f'E[U] with perfect all X_e: {expected_utility_perfect_info_all/1e6:.2f}')
plt.axvline(expected_utilities_uncertain[optimal_decision_uncertain]/1e6, color='red', linestyle='--', 
           label=f'E[U] under uncertainty: {expected_utilities_uncertain[optimal_decision_uncertain]/1e6:.2f}')

plt.xlabel('Utility (million euro)')
plt.ylabel('Density')
plt.title('Comparison of Utilities with Different Information Scenarios')
plt.legend()
plt.grid()
plt.show()

# Summary of Value of Information results
print(f"\n=== VALUE OF INFORMATION SUMMARY ===")
print(f"VPI for M only: {value_of_perfect_information_M / 1e6:.2f} million euro")
print(f"evpm for all X_e: {evpm / 1e6:.2f} million euro")
print(f"Additional value from knowing R and CF (beyond M): {(evpm - value_of_perfect_information_M) / 1e6:.2f} million euro")

# Sample information value for M
# Z = sum of exp(-S_i)

# Use the same baseline calculation as before for consistency
EU_baseline = expected_utilities_uncertain
a_opt = np.argmax(EU_baseline)
EU_current = np.max(EU_baseline)


n_s_list = np.array([1, 5, 10, 20, 50, 100])
VZ = []
for n_s in n_s_list:
    # simulate data samples conditional on M
    S_samps = gumbel_r.rvs(loc = M_samps[:, None], scale=beta, size=(N, n_s), random_state=42)
    Z_samps = np.sum(np.exp(-S_samps), axis=1)

    # approximate posterior weights p(M|Z) ∝ p(Z|M)p(M)
    # Here we just use p(Z|M) as weights because M prior is normal
    # Using log-likelihood from Eq (22)
    # L(m|Z) ∝ exp( n_s * ( -Σs_i + m - exp(-s_i + m) ) )
    # but Σs_i depends only on Z, we can simplify via:
    # L ∝ exp( -Z * exp(m) + n_s * m )
    logw = n_s * M_samps - np.exp(M_samps) * Z_samps
    logw -= np.max(logw)  # numerical stability
    w = np.exp(logw)
    w /= np.sum(w)

    # compute expected utility under posterior weights
    EU_post = np.zeros(3)
    for i in range(3):
        # Calculate utility for each sample, then take weighted average
        utilities = -compute_Ye(M_samps, R_samples[i], CF_samps) - ca_vals[i]
        EU_post[i] = np.sum(w * utilities)

    EU_opt_post = np.max(EU_post)
    VZ.append(EU_opt_post - EU_current)
    
VZ = np.array(VZ)

# --- plot result ---
plt.figure(figsize=(8,5))
plt.plot(n_s_list, VZ/1e6, 'o-', color='#3c649f')
plt.xlabel('Sample size n_s')
plt.ylabel('Sample information value V_Z (million €)')
plt.title('Sample information value vs number of samples n_s')
plt.grid(True)
plt.show()