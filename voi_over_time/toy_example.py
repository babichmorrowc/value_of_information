# Toy example to explore BDA & VoI over time

import numpy as np
import matplotlib.pyplot as plt

# Number of time steps
T = 50
# Lambda
lambda_val = 1

# Simulate state of nature theta
# The mean of theta grows exponentially over time
# The standard deviation of theta grows exponentially over time
def simulate_theta(T, n_trajectories=100):
    time = np.arange(T)
    mean_theta = np.exp(0.05 * time)  # Exponential growth of mean
    std_theta = 0.1 * np.exp(0.05 * time)  # Exponential growth of std
    # Sample n full trajectories for theta using the time-dependent distributions
    theta = np.random.normal(loc=mean_theta, scale=std_theta, size=(n_trajectories, T))
    return theta

# Simulate theta over T time steps
theta_T = simulate_theta(T)  # (n_trajectories x T)
# Plot the simulated theta trajectories over time
plt.figure(figsize=(12, 6))
for n in range(theta_T.shape[0]):
    plt.plot(np.arange(theta_T.shape[1]), theta_T[n], alpha=0.5)
plt.title('Simulated Theta trajectories over time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid()
plt.show()

# Define the inputs X
# X_decision = [C_0, C_1, O_0, O_1, B_0, B_1]
X_decision = np.array([0, 10, 0, 1, 0, 0.2])

# Define the loss function
# Loss at time t of making decision d at time t_star
def loss_function(theta_t, X_decision, d, lambda_val, t_star, t):
    if t < t_star:
        return lambda_val * theta_t
    else:
        operational_cost = X_decision[2 + d]
        upfront_cost = X_decision[d]
        benefit = X_decision[4 + d]
        ongoing_loss = operational_cost + (1 - benefit) * lambda_val * theta_t
        if t == t_star:
            return upfront_cost + ongoing_loss
        else:
            return ongoing_loss

# Define the utility function
