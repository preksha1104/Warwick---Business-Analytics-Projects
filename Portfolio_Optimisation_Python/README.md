The report conducts a portfolio optimization analysis using three models: Mean-Variance, Minimum-Variance, and Naïve models. The goal is to evaluate risk-return dynamics for portfolios categorized by firm size and Book-to-Market (B/M) ratios.

Problem Being Tackled:
The study aims to optimize investment portfolios by balancing risk and return using different portfolio construction techniques. Specifically, it investigates:

1. How different firm sizes (Small vs. Big) and B/M ratios (Growth, Neutral, Value) impact portfolio performance.
2. Which portfolio optimization model (Mean-Variance, Minimum-Variance, or Naïve) provides the best trade-off between risk and return?
3. How different time windows (12, 36, and 60 months) affect portfolio stability and risk-adjusted returns?
4. Whether optimization models outperform a simple equally weighted (Naïve) strategy in achieving better returns at lower risk.

EDA Analysis:

1. Value-weighted returns used for better comparison with major indices.
2. Histogram analysis to assess volatility and return distribution.
3. Trend analysis to study portfolio behavior during market crises (Dot-com Bubble, 2008 Financial Crisis, COVID-19).

Mathematical Formulations:

1. Rolling Windows (12, 36, 60 months): Used to estimate mean and covariance matrix for portfolio optimization.
2. Capital Asset Pricing Model (CAPM): Used to set target return constraints in the Mean-Variance optimization.
3. Beta Analysis: Measures portfolio risk relative to the market (MSCI World Index).

Performance Evaluation Metrics:

1. Efficient Frontier & Capital Market Line (CML): To visualize the risk-return trade-off.
2. Mean-Standard Deviation Diagrams: To compare portfolio risk and return.
3. Sharpe Ratio: Evaluates risk-adjusted returns.
4. Optimal Weight Allocation: Compares portfolio composition across models.

Key Insights:

Impact of Rolling Window Selection:

12-month window: Captures short-term market shifts and yields higher risk-adjusted returns.
36-month window: Balances market fluctuations and broader trends.
60-month window: Less stable, sometimes underperforms naïve models, indicating a need for further model refinement.

Naïve vs. Optimization Models:

Optimization models outperformed the Naïve strategy in all cases, offering higher returns with lower risk.
Minimum-Variance (12-month) provided the best risk-adjusted performance.
Mean-Variance (60-month) showed inconsistent results, highlighting stability concerns.

Conclusion:
Short-term (12-month window): Minimum-Variance performed best, offering higher returns at the lowest risk.
Medium-term (36-month window): Mean-Variance optimization outperformed the Naïve model, but with declining Sharpe Ratios.
Long-term (60-month window): Performance declined, showing model stability issues.

Overall, sophisticated optimization techniques (Mean-Variance & Minimum-Variance) significantly outperformed the Naïve model in both return and risk management.
The study emphasizes the importance of continuous refinement and evaluation of portfolio optimization strategies to adapt to evolving market conditions.