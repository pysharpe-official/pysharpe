🧮 PySharpe — Simple, Pandas-Based Portfolio Analytics

PySharpe is a lightweight, open-source portfolio performance analysis library for Python, built entirely on top of pandas and numpy. It enables quantitative investors, researchers, and analysts to easily compute key performance metrics, visualize aggregated returns, and export professional-grade Excel performance reports — all with minimal code.

🚀 Features

Plug-and-Play Analysis: Input a pandas.DataFrame or Series of monthly returns.

Comprehensive Metrics:

CAGR, Volatility, Downside Deviation

Sharpe, Sortino, Beta, Alpha

Max & Average Drawdown, Recovery Time

VaR / CVaR (Expected Shortfall)

Recovery-Scaled Sharpe (RSS)

Benchmark Comparison: Evaluate strategy returns vs. benchmark (e.g., market index).

Monthly and Yearly Aggregation: Easily group and visualize average monthly or yearly returns.

Excel Dashboard Builder:

Automatically generates a full Excel report with formatted tables,
performance charts, and growth curves ($100 starting value).

Conditional formatting highlights outperformance vs. benchmark.

Visualization-Ready: Built-in support for matplotlib and plotly.

📦 Installation

pip install pysharpe


(Coming soon — once the package is published on PyPI)

For now, clone the repo:

git clone https://github.com/pysharpe-official/pysharpe.git
cd pysharpe
pip install -r requirements.txt


🧠 Quick Start

import pandas as pd
from pysharpe import QPySharpePanalysis

# Example DataFrame
df = pd.DataFrame({
    'date': pd.date_range(start='2018-01-01', periods=48, freq='M'),
    'weighted_month_return': np.random.normal(0.01, 0.03, 48),
    'market_return': np.random.normal(0.008, 0.025, 48)
})

analyzer = pysharpe_basic(df, rf_rate=0.02)

# Compute metrics
metrics = analyzer._analyze()
print(metrics)

# Create full Excel dashboard
analyzer.full_excel_report("portfolio_analysis.xlsx")


📊 Example Output

Sheet 1: Summary metrics (CAGR, Sharpe, Alpha, etc.)
Sheet 2: Average monthly returns with bar chart
Sheet 3: Yearly and monthly performance history
Sheet 4: Growth-of-$100 chart (Strategy vs Benchmark)

🧩 Dependencies

pandas
numpy
matplotlib
xlsxwriter
plotly

🧱 Project Goals

PySharpe is designed to:

Simplify quantitative performance analysis for everyday investors
Offer a clean pandas-based API with transparent math
Generate ready-to-share Excel performance reports
Remain fully open-source, extensible, and community-driven

🤝 Contributing

Pull requests are welcome!
If you’d like to add features (like rolling stats, drawdown visualizations, or Monte Carlo simulations), fork the repo and open a PR.

📜 License

MIT License © 2025 pysharpe
