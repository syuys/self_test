import numpy as np

# Set parameters
days = 100                 # Number of days to simulate
simulations = 1000         # Number of simulation runs
prob_gain = 0.51           # Probability of gain
gain = 0.01                # Gain rate (+1%)
loss = -0.01               # Loss rate (-1%)


########### Monte carlo simulated Sharpe Ratio ############
# Fix seed and Simulate daily returns
np.random.seed(1)
returns = np.random.choice([gain, loss], size=(simulations, days), p=[prob_gain, 1 - prob_gain])

# Calculate mean returns for each simulation
mean_returns = np.mean(returns, axis=1)

# Calculate the sample mean and sample standard deviation of the mean returns
sample_mean = np.mean(mean_returns)
sample_std = np.std(mean_returns, ddof=1)

# Calculate Sharpe Ratio using the sampling method
sharpe_ratio_sampling = sample_mean / sample_std

# Display the results
print(f"Sharpe Ratio (Monte-Carlo sampling): {sharpe_ratio_sampling:.4f}")  # 0.1923


########### Expected Sharpe Ratio ############
e_of_x = loss * (1 - prob_gain) + gain * prob_gain                 # E(X)
e_of_x_square = loss**2 * prob_gain + gain**2 * (1 - prob_gain)    # E(X^2)
variance = e_of_x_square - e_of_x**2                               # Var(X) = E(X^2) - E(X)^2
variance_multiple_days = variance / days                           # Variance for multiple days
std_multiple_days = np.sqrt(variance_multiple_days)                # Standard deviation for multiple days
e_of_sharpe = e_of_x / std_multiple_days
print(f"Expected Sharpe Ratio: {e_of_sharpe:.4f}")                 # 0.2
