# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:45:35 2024

"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
 
# Load the data
file_path = 'Developed_value_weighted_return_monthly.csv'
portfolios_df = pd.read_csv(file_path)
 
# Rename columns for better readability
portfolios_df.columns = [
    'Date', 'Small_Growth', 'Small_Neutral', 'Small_Value', 
    'Big_Growth', 'Big_Neutral', 'Big_Value'
]
 
# Remove the first row which contains column descriptions
portfolios_df = portfolios_df.drop(0)
 
# Convert the 'Date' column to a datetime format
portfolios_df['Date'] = pd.to_datetime(portfolios_df['Date'], format='%Y%m')
 
# Set 'Date' as the index
portfolios_df.set_index('Date', inplace=True)
 
# Convert the remaining columns to numeric
portfolios_df = portfolios_df.apply(pd.to_numeric)

# check no missing values
missing_values = portfolios_df.isnull().sum()

# Summary statistics
print("Summary Statistics:")
summary_statistics = portfolios_df.describe()

# Plot histograms and KDE plots
fig, axs = plt.subplots(2, 3, figsize=(30, 20))
colors = ['blue', 'orange', 'green', '#333333', 'purple', 'red']

for i, column in enumerate(portfolios_df.columns):
    row = i // 3
    col = i % 3
    sns.histplot(portfolios_df[column], kde=True, ax=axs[row, col], color=colors[i], bins=50)
    axs[row, col].set_title(column, fontsize=24)  # Larger title font size
    axs[row, col].set_xlabel('Return', fontsize=20)  # X-axis label
    axs[row, col].set_ylabel('Frequency', fontsize=20)  # Y-axis label
    axs[row, col].tick_params(axis='x', labelsize=18)  # X-axis tick labels
    axs[row, col].tick_params(axis='y', labelsize=18)  # Y-axis tick labels
    

plt.tight_layout()
plt.show()

# Plot firm size to compare market cap

avg_firm_size_df = pd.read_csv('Developed_firm_size.csv')

# Rename columns for better readability
avg_firm_size_df.columns = [
    'Date', 'Small_Growth', 'Small_Neutral', 'Small_Value', 
    'Big_Growth', 'Big_Neutral', 'Big_Value'
]
 
# Remove the first row which contains column descriptions
avg_firm_size_df = avg_firm_size_df.drop(0)
 
# Convert the 'Date' column to a datetime format
avg_firm_size_df['Date'] = pd.to_datetime(avg_firm_size_df['Date'], format='%Y%m')
 
# Set 'Date' as the index
avg_firm_size_df.set_index('Date', inplace=True)
 
# Convert the remaining columns to numeric
avg_firm_size_df = avg_firm_size_df.apply(pd.to_numeric)


plt.figure(figsize=(14, 6))

# Plot for Small cap
plt.subplot(1, 2, 1)
for column in ['Small_Growth', 'Small_Neutral', 'Small_Value']:  
    plt.plot(avg_firm_size_df.index, avg_firm_size_df[column], label=column)
plt.xlabel('Date')
plt.ylabel('Average Firm Size')
plt.title('Average Firm Size of Small Portfolios Over Time')
plt.legend()
plt.grid(True)

# Plot for Big cap
plt.subplot(1, 2, 2)
for column in ['Big_Growth', 'Big_Neutral', 'Big_Value']:  
    plt.plot(avg_firm_size_df.index, avg_firm_size_df[column], label=column)
plt.xlabel('Date')
plt.ylabel('Average Firm Size')
plt.title('Average Firm Size of Big Portfolios Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot average firm count 

avg_firm_count_df = pd.read_csv('Developed_no._firm.csv')

# Rename columns for better readability
avg_firm_count_df.columns = [
    'Date', 'Small_Growth', 'Small_Neutral', 'Small_Value', 
    'Big_Growth', 'Big_Neutral', 'Big_Value'
]
 
# Remove the first row which contains column descriptions
avg_firm_count_df = avg_firm_count_df.drop(0)
 
# Convert the 'Date' column to a datetime format
avg_firm_count_df['Date'] = pd.to_datetime(avg_firm_count_df['Date'], format='%Y%m')
 
# Set 'Date' as the index
avg_firm_count_df.set_index('Date', inplace=True)
 
# Convert the remaining columns to numeric
avg_firm_count_df = avg_firm_count_df.apply(pd.to_numeric)

# plot for trend of firms in each portfolio over time

plt.figure(figsize=(12, 8))

# Plot for Small portfolios
plt.subplot(2, 1, 1)
for column in avg_firm_count_df.columns[:3]:  # Considering first 3 columns are related to Small portfolios
    plt.plot(avg_firm_count_df.index, avg_firm_count_df[column], label=column, alpha=0.8)  # Adjust alpha value as needed

plt.xlabel('Date')
plt.ylabel('Number of Firms')
plt.title('Number of Firms in Small Portfolios Over Time')
plt.legend()
plt.grid(True)

# Plot for Big portfolios
plt.subplot(2, 1, 2)
for column in avg_firm_count_df.columns[3:]:  # Considering last 3 columns are related to Big portfolios
    plt.plot(avg_firm_count_df.index, avg_firm_count_df[column], label=column, alpha=0.8)  # Adjust alpha value as needed

plt.xlabel('Date')
plt.ylabel('Number of Firms')
plt.title('Number of Firms in Big Portfolios Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()

#Plotting the correlation matrix
corr_matrix = portfolios_df.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, linewidths=0.5)
plt.title('Correlation Matrix of Portfolios')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

#Plot the returns over time

# Define the figure and axes
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Define a list of colors to use for each asset
colors = ['blue', 'orange', 'green', '#333333', 'purple', 'red']

# Iterate through each column (asset) and plot with a different color
for i, (column, color) in enumerate(zip(portfolios_df.columns[0:], colors)):
    row = i // 3
    col = i % 3
    axs[row, col].plot(portfolios_df.index, portfolios_df[column], color=color)
    axs[row, col].set_title(column)
    axs[row, col].set_xlabel('Date')
    axs[row, col].set_ylabel('Return')

# Adjust layout
plt.tight_layout()
plt.show()

# Step 1: Calculate expected returns and covariance matrix

expected_returns = portfolios_df.mean(axis=0)
covariance_matrix = np.cov(portfolios_df.T)

# Step 2: Generate random portfolios
np.random.seed(42)
num_portfolios = 1000
num_assets = len(expected_returns)
random_weights = np.random.dirichlet(np.ones(num_assets), size=num_portfolios)

# Step 3: Calculate expected return and volatility for each random portfolio
portfolio_returns = np.dot(random_weights, expected_returns)
portfolio_volatility = np.sqrt(np.array([np.dot(np.dot(weights.T, covariance_matrix), weights) for weights in random_weights]))

# Step 4: Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_volatility, portfolio_returns, c=portfolio_returns / portfolio_volatility, marker='o', cmap='viridis')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)

# Step 5: Identify and mark minimum variance point
min_variance_idx = np.argmin(portfolio_volatility)
plt.scatter(portfolio_volatility[min_variance_idx], portfolio_returns[min_variance_idx], marker='x', color='red', label='Minimum Variance')

# Step 6: Show plot
plt.legend()
plt.show()

# Assuming rf is the risk-free rate
rf = 4.13/12 # 4.13% pa risk free return

# Calculate Sharpe ratios for each portfolio
sharpe_ratios = (portfolio_returns - rf) / portfolio_volatility

# Plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_volatility, portfolio_returns, c=sharpe_ratios, marker='o', cmap='viridis', label='Portfolios')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier and Capital Market Line')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)

# Identify the tangency portfolio
tangency_idx = np.argmax(sharpe_ratios)
tangency_return = portfolio_returns[tangency_idx]
tangency_volatility = portfolio_volatility[tangency_idx]

# Plot Capital Market Line
cml_x = np.linspace(0, max(portfolio_volatility), 100)
cml_y = rf + (tangency_return - rf) / tangency_volatility * cml_x
plt.plot(cml_x, cml_y, color='red', linestyle='--', label='Capital Market Line')

# Mark Tangency Portfolio
plt.scatter(tangency_volatility, tangency_return, marker='x', color='red', s=100, label='Tangency Portfolio')

# Adjust layout
plt.legend()
plt.show()

# Calculate rolling mean and covariance
def rolling_mean_cov(df, window):
    mean_returns = df.rolling(window).mean().shift(1)
    cov_matrices = df.rolling(window).cov().shift(1)
    return mean_returns, cov_matrices
 
# Calculate rolling mean and covariance for different windows
mean_12, cov_12 = rolling_mean_cov(portfolios_df, 12)
mean_36, cov_36 = rolling_mean_cov(portfolios_df, 36)
mean_60, cov_60 = rolling_mean_cov(portfolios_df, 60)

# Set target return using CAPM method
# Assumptions for CAPM
risk_free_rate = 4.13/12  # 4.13% annual risk-free rate
market_return = 9.71/12 # 9.71% annual market return
 
# Calculate beta for each portfolio
market_index = portfolios_df.mean(axis=1) 
betas = portfolios_df.apply(lambda x: np.cov(x, market_index)[0, 1] / np.var(market_index))
 
# Calculate target returns using CAPM
target_returns = risk_free_rate + (betas * (market_return - risk_free_rate))
 
# Ensure target_returns index matches portfolios_df index
target_returns = pd.Series(target_returns, index=portfolios_df.columns)

# Define colors based on beta value
colors = ['green' if beta > 1 else 'red' for beta in betas]

# Plotting
plt.scatter(betas, portfolios_df.columns[0:], color=colors)
plt.axvline(x=1, color='gray', linestyle='--', label='Market Beta = 1')
plt.xlabel('Beta')
plt.ylabel('Asset')
plt.title('Beta of Assets')
plt.legend()
plt.show()

#Mean-variance optimisation

# Define the function to calculate portfolio performance
def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.dot(weights, returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility
 
# Define the function to minimize (volatility)
def minimize_volatility(weights, returns, cov_matrix):
    return portfolio_performance(weights, returns, cov_matrix)[1]
 
# Define the optimization function
def optimize_portfolio(mean_returns, cov_matrix, target_return):
    num_assets = len(mean_returns)
    # Constraints: sum of weights is 1 and portfolio return is target_return
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target_return}
    )
    # Bounds for weights
    bounds = tuple((0, 1) for _ in range(num_assets))
    # Initial guess for weights
    initial_guess = num_assets * [1. / num_assets]
    # Optimize the portfolio
    result = minimize(minimize_volatility, initial_guess, args=(mean_returns, cov_matrix), 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result
 
# Initialize dictionary to store results for each window size
results = {}
 
# Perform optimization for each window size
for window_size, mean_returns, cov_matrices in zip([12, 36, 60], [mean_12, mean_36, mean_60], [cov_12, cov_36, cov_60]):
    optimal_returns = []
    optimal_volatilities = []
    optimal_weights = []
    mean_returns_list = []
    cov_matrices_list = []
    dates = []
    next_day_returns = []
    current_day_returns = []
 
    for i in range(len(mean_returns.index[window_size:]) - 1):
        date = mean_returns.index[window_size:][i]
        date_next_day = mean_returns.index[window_size:][i+1]
        mean_ret = mean_returns.loc[date]
        cov_mat = cov_matrices.loc[date]
        if not mean_ret.isnull().any() and not cov_mat.isnull().any().any():
            target_ret = target_returns.mean()  # Using mean of target returns as target return for optimization
            result = optimize_portfolio(mean_ret, cov_mat, target_ret)
            optimal_return, optimal_volatility = portfolio_performance(result.x, mean_ret, cov_mat)
            optimal_returns.append(optimal_return)
            optimal_volatilities.append(optimal_volatility)
            optimal_weights.append(result.x)
            mean_returns_list.append(mean_ret.values)
            cov_matrices_list.append(cov_mat.values)
            dates.append(date)
            current_day_return = np.dot(result.x, portfolios_df.loc[date])
            current_day_returns.append(current_day_return)
            # Calculate rebalanced return for the next day if date_next_day exists in portfolios_df index
        if date_next_day in portfolios_df.index:
            return_next_day = np.dot(result.x, portfolios_df.loc[date_next_day])
            next_day_returns.append(return_next_day)
        else:
            print(f"Warning: Date {date_next_day} does not exist in portfolios_df index.")
            next_day_returns.append(np.nan)  # Append NaN for missing data
  
    next_day_returns_shifted = [np.nan] + next_day_returns[:-1]

    
   # Create DataFrame for results
    results_df = pd.DataFrame({
        'Date': dates,
        'Optimal Return': optimal_returns,
        'Optimal Volatility': optimal_volatilities,
        'Optimal Weights': optimal_weights,
        'Mean Returns': mean_returns_list,
        'Covariance Matrix': cov_matrices_list,
        'Current day return' : current_day_returns,
        'Next day return' : next_day_returns_shifted
    })
    
    # Add columns for each asset's weight
    for i, asset in enumerate(portfolios_df.columns):
        results_df[asset + '_Weight'] = results_df['Optimal Weights'].apply(lambda x: x[i])
    # Set the 'Date' column as the index
    results_df.set_index('Date', inplace=True)
    # Store the results for the current window size
    results[window_size] = results_df
 
# Display the results for each window size
for window_size, df in results.items():
    print(f"Results for window size {window_size} months:")
    print(df.head(), "\n")
    # Check if the sum of weights equals 1 for each set of optimal weights
sum_of_weights = df['Optimal Weights'].apply(lambda x: np.sum(x))
print("Check if sum of weights equals 1:")
print(sum_of_weights)



# If you need to access individual dataframes:
results_12 = results[12]
results_36 = results[36]
results_60 = results[60]

# Initialize an empty dictionary to store mean Sharpe ratios for each window size
mean_sharpe_ratios = {}

# Calculate Sharpe ratios for each window size
for window_size, df in results.items():
    excess_returns = df['Optimal Return'] - risk_free_rate
    volatility_excess = df['Optimal Volatility']
    sharpe_ratio = excess_returns / volatility_excess
    mean_sharpe_ratio = sharpe_ratio.mean()
    mean_sharpe_ratios[window_size] = mean_sharpe_ratio

# Convert the dictionary into a dataframe
mean_sharpe_df = pd.DataFrame(list(mean_sharpe_ratios.items()), columns=['Window Size (Months)', 'Mean Sharpe Ratio'])

# Display the dataframe
print(mean_sharpe_df)

# Convert data to DataFrame
weights_dfs = [results_12.iloc[:, -6:], results_36.iloc[:, -6:], results_60.iloc[:, -6:]]
windows = [12, 36, 60]
 
# Iterate over each window size and corresponding DataFrame
for window, weights_df in zip(windows, weights_dfs):
    # Define the figure and subplots
    fig, axs = plt.subplots(3, 2, figsize=(16, 10))
 
    # Define a list of colors to use for each asset
    colors = ['blue', 'orange', 'green', '#333333', 'purple', 'red']
 
    # Iterate through each asset and plot on a different subplot
    for ax, (column, color) in zip(axs.flatten(), zip(weights_df.columns, colors)):
        ax.scatter(weights_df.index, weights_df[column], label=column, color=color, s=10)
        ax.set_title(column)
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.legend()
 
    # Adjust layout
    plt.tight_layout()
    plt.suptitle(f'Window Size: {window} months', fontsize=16)
    plt.show()
    


# naive model
def naive_model(returns):
    num_assets = returns.shape[1]
    naive_weights = np.ones(num_assets) / num_assets
    portfolio_return = np.dot(returns, naive_weights)  # Calculate portfolio return
    
    
    # Create DataFrame to store results
    result_naive = pd.DataFrame({
        'Date': returns.index,
        'Portfolio Return': portfolio_return,

    })
    
    return result_naive

# Apply naive model to your dataset
result_naive = naive_model(portfolios_df)


# Create subplots for each window size
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Plot next-day returns of MPT for each window size
axs[0].plot(results_12.index, results_12['Next day return'], label='MPT (Window Size: 12 months)', color='blue')
axs[1].plot(results_36.index, results_36['Next day return'], label='MPT (Window Size: 36 months)', color='green')
axs[2].plot(results_60.index, results_60['Next day return'], label='MPT (Window Size: 60 months)', color='orange')

# Plot portfolio returns of naive model for each window size
axs[0].plot(result_naive['Date'], result_naive['Portfolio Return'], label='Naive Model Portfolio Return', color='red')
axs[1].plot(result_naive['Date'], result_naive['Portfolio Return'], label='Naive Model Portfolio Return', color='red')
axs[2].plot(result_naive['Date'], result_naive['Portfolio Return'], label='Naive Model Portfolio Return', color='red')

# Add labels and title for each subplot
for i, ax in enumerate(axs):
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.set_title(f'Comparison of Next-day Returns (MPT vs. Naive Model) - Window Size: {12 * (i+1)} months')
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


#min_var


# Define the optimization function
def optimize_portfolio_min_var(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    # Constraints: sum of weights is 1 
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        )
    # Bounds for weights
    bounds = tuple((0, 1) for _ in range(num_assets))
    # Initial guess for weights
    initial_guess = num_assets * [1. / num_assets]
    # Optimize the portfolio
    result_min_var = minimize(minimize_volatility, initial_guess, args=(mean_returns, cov_matrix), 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result_min_var

# Initialize dictionary to store results for each window size
results_min_var = {}
 
# Perform optimization for each window size
for window_size, mean_returns, cov_matrices in zip([12, 36, 60], [mean_12, mean_36, mean_60], [cov_12, cov_36, cov_60]):
    optimal_returns = []
    optimal_volatilities = []
    optimal_weights = []
    mean_returns_list = []
    cov_matrices_list = []
    dates = []
    next_day_returns = []
    current_day_returns = []
 
    for i in range(len(mean_returns.index[window_size:]) - 1):
        date = mean_returns.index[window_size:][i]
        date_next_day = mean_returns.index[window_size:][i+1]
        mean_ret = mean_returns.loc[date]
        cov_mat = cov_matrices.loc[date]
        if not mean_ret.isnull().any() and not cov_mat.isnull().any().any():
            
            result_min_var = optimize_portfolio_min_var(mean_ret, cov_mat)
            optimal_return, optimal_volatility = portfolio_performance(result_min_var.x, mean_ret, cov_mat)
            optimal_returns.append(optimal_return)
            optimal_volatilities.append(optimal_volatility)
            optimal_weights.append(result_min_var.x)
            mean_returns_list.append(mean_ret.values)
            cov_matrices_list.append(cov_mat.values)
            dates.append(date)
            current_optimal_return = np.dot(result_min_var.x, portfolios_df.loc[date])
            current_day_returns.append(current_optimal_return)
            # Calculate rebalanced return for the next day if date_next_day exists in portfolios_df index
        if date_next_day in portfolios_df.index:
            return_next_day = np.dot(result_min_var.x, portfolios_df.loc[date_next_day])
            next_day_returns.append(return_next_day)
        else:
            print(f"Warning: Date {date_next_day} does not exist in portfolios_df index.")
            next_day_returns.append(np.nan)  # Append NaN for missing data
  
    next_day_returns_min_var = [np.nan] + next_day_returns[:-1]

    
   # Create DataFrame for results
    results_df_min_var = pd.DataFrame({
        'Date': dates,
        'Optimal Return': optimal_returns,
        'Optimal Volatility': optimal_volatilities,
        'Optimal Weights': optimal_weights,
        'Mean Returns': mean_returns_list,
        'Covariance Matrix': cov_matrices_list,
        'Current day return' : current_day_returns,
        'Next day return' : next_day_returns_min_var
    })
    # Add columns for each asset's weight
    for i, asset in enumerate(portfolios_df.columns):
        results_df_min_var[asset + '_Weight'] = results_df_min_var['Optimal Weights'].apply(lambda x: x[i])
    # Set the 'Date' column as the index
    results_df_min_var.set_index('Date', inplace=True)
    # Store the results for the current window size
    results_min_var[window_size] = results_df_min_var
 
# Display the results for each window size
for window_size, df in results_min_var.items():
    print(f"Results for window size {window_size} months:")
    print(df.head(), "\n")
    # Check if the sum of weights equals 1 for each set of optimal weights
sum_of_weights = df['Optimal Weights'].apply(lambda x: np.sum(x))
print("Check if sum of weights equals 1:")
print(sum_of_weights)



# If you need to access individual dataframes:
results_12_min = results_min_var[12]
results_36_min = results_min_var[36]
results_60_min = results_min_var[60]

# Initialize an empty dictionary to store mean Sharpe ratios for each window size
mean_sharpe_ratios_min_var = {}


# Calculate Sharpe ratios for each window size
for window_size, df in results_min_var.items():
    excess_returns = df['Optimal Return'] - risk_free_rate
    volatility_excess = df['Optimal Volatility']
    sharpe_ratio = excess_returns / volatility_excess
    mean_sharpe_ratio = sharpe_ratio.mean()
    mean_sharpe_ratios_min_var[window_size] = mean_sharpe_ratio

# Convert the dictionary into a dataframe
mean_sharpe_df_min_var = pd.DataFrame(list(mean_sharpe_ratios_min_var.items()), columns=['Window Size (Months)', 'Mean Sharpe Ratio'])

# Display the dataframe
print(mean_sharpe_df_min_var)

# Convert data to DataFrame
weights_dfs = [results_12_min.iloc[:, -6:], results_36_min.iloc[:, -6:], results_60_min.iloc[:, -6:]]
windows = [12, 36, 60]
 
# Iterate over each window size and corresponding DataFrame
for window, weights_df in zip(windows, weights_dfs):
    # Define the figure and subplots
    fig, axs = plt.subplots(3, 2, figsize=(16, 10))
 
    # Define a list of colors to use for each asset
    colors = ['blue', 'orange', 'green', '#333333', 'purple', 'red']
 
    # Iterate through each asset and plot on a different subplot
    for ax, (column, color) in zip(axs.flatten(), zip(weights_df.columns, colors)):
        ax.scatter(weights_df.index, weights_df[column], label=column, color=color, s=10)
        ax.set_title(column)
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.legend()
 
    # Adjust layout
    plt.tight_layout()
    plt.suptitle(f'Window Size: {window} months', fontsize=16)
    plt.show()

# Create subplots for each window size
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Plot next-day returns of Min-Var for each window size
axs[0].plot(results_12_min.index, results_12_min['Next day return'], label='Min-Var (Window Size: 12 months)', color='blue')
axs[1].plot(results_36_min.index, results_36_min['Next day return'], label='Min-Var (Window Size: 36 months)', color='green')
axs[2].plot(results_60_min.index, results_60_min['Next day return'], label='Min-Var (Window Size: 60 months)', color='orange')

# Plot portfolio returns of naive model for each window size
axs[0].plot(result_naive['Date'], result_naive['Portfolio Return'], label='Naive Model Portfolio Return', color='red')
axs[1].plot(result_naive['Date'], result_naive['Portfolio Return'], label='Naive Model Portfolio Return', color='red')
axs[2].plot(result_naive['Date'], result_naive['Portfolio Return'], label='Naive Model Portfolio Return', color='red')

# Add labels and title for each subplot
for i, ax in enumerate(axs):
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.set_title(f'Comparison of Next-day Returns (Min-Var vs. Naive Model) - Window Size: {12 * (i+1)} months')
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


# Calculate naive policy metrics

naive_policy_return = portfolios_df.mean(axis=1).mean()  # Average return of all portfolios
naive_policy_volatility = portfolios_df.mean(axis=1).std()  # Volatility of the average returns


# Calculate excess returns
naive_excess_returns = naive_policy_return - risk_free_rate

# Calculate Sharpe ratio
naive_sharpe_ratio = naive_excess_returns / naive_policy_volatility

# Initialize lists to store aggregated values
optimal_return_means = []
optimal_volatility_means = []
 
# Iterate over each dataframe and calculate the mean optimal return and volatility
for df in [results_12, results_36, results_60, results_12_min, results_36_min, results_60_min]:
    optimal_return_mean = df['Optimal Return'].mean()
    optimal_volatility_mean = df['Optimal Volatility'].mean()
    # Append means to the lists
    optimal_return_means.append(optimal_return_mean)
    optimal_volatility_means.append(optimal_volatility_mean)
 
# Append naive policy metrics to the lists


optimal_return_means.append(naive_policy_return)
optimal_volatility_means.append(naive_policy_volatility)
 
# Create a summary DataFrame
summary_df = pd.DataFrame({
    'Window Size': [12, 36, 60, 12, 36, 60, 'N/A'],  # Repeating window sizes for both optimal return and minimum variance, plus one for naive
    'Method': ['mean_variance', 'mean_variance', 'mean_variance', 'min_variance', 'min_variance', 'min_variance', 'naive'],
    'Optimal Return Mean': optimal_return_means,
    'Optimal Volatility Mean': optimal_volatility_means
})
 
# Display the summary DataFrame
print(summary_df)

# Create scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=summary_df, x='Optimal Return Mean', y='Optimal Volatility Mean', hue='Method', style='Window Size', s=100)

# Add title and labels
plt.title('Comparison of Optimization Strategies by Mean Return and Volatility')
plt.xlabel('Mean Optimal Return')
plt.ylabel('Mean Optimal Volatility')

# Show plot
plt.legend(title='Method / Window Size')
plt.grid(True)
plt.show()


