# Toy example to explore BDA & VoI over time

import numpy as np
import matplotlib.pyplot as plt

# Simulate state of nature theta
# The mean of theta grows exponentially over time
# The standard deviation of theta grows exponentially over time
def simulate_theta(T):
    time = np.arange(T)
    mean_theta = np.exp(0.05 * time)  # Exponential growth of mean
    std_theta = 0.1 * np.exp(0.05 * time)  # Exponential growth of std
    theta = np.random.normal(mean_theta, std_theta)
    return theta, mean_theta, std_theta

# Simulate theta over 10 time steps
theta_10 = simulate_theta(10)