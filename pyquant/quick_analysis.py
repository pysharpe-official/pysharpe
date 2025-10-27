import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import plotly.express as px
import plotly.io as pio
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class QPyQuantPanalysis:
    """
    Comprehensive Portfolio Performance Analyzer for Quant Strategies.
    ------------------------------------------------------------------
    Expects a DataFrame with columns:
        - 'date'
        - 'weighted_month_return'  (strategy monthly returns)
        - 'market_return'          (benchmark monthly returns)
    """

    def __init__(self, df: pd.DataFrame, rf_rate: float):
        """Initialize analyzer with strategy/benchmark monthly returns and risk-free rate."""
        self.df = df.copy()
        self.rf_rate = rf_rate
        self.df['strategy'] = self.df['weighted_month_return']
        self.df['benchmark'] = self.df['market_return']
        self.df = self.df.dropna().reset_index(drop=True)

    # ================================================================
    # ===                   STATISTICAL FUNCTIONS                  ===
    # ================================================================

    # --------------------------------------------------------
    # ---------- Helper: CAGR (Compound Annual Growth) --------
    def _cagr(self, returns):
        """Compound Annual Growth Rate (CAGR)."""
        total_return = (1 + returns).prod()
        n_months = len(returns)
        return total_return ** (12 / n_months) - 1

    # --------------------------------------------------------
    # ---------- Helper: Annualized Volatility ---------------
    def _volatility(self, returns):
        """Annualized volatility (standard deviation of returns)."""
        return returns.std() * np.sqrt(12)

    # --------------------------------------------------------
    # ---------- Helper: Downside Deviation ------------------
    def _downside_deviation(self, returns):
        """Annualized standard deviation of negative (downside) returns."""
        downside = returns[returns < 0]
        return downside.std() * np.sqrt(12)

    # --------------------------------------------------------
    # ---------- Helper: Sharpe Ratio ------------------------
    def _sharpe_ratio(self, returns):
        """Risk-adjusted return using total volatility."""
        excess = returns - (self.rf_rate / 12)
        return np.mean(excess) / np.std(returns) * np.sqrt(12)

    # --------------------------------------------------------
    # ---------- Helper: Sortino Ratio -----------------------
    def _sortino_ratio(self, returns):
        """Risk-adjusted return using downside deviation relative to the risk-free rate."""
        mar = self.rf_rate / 12  # minimum acceptable return per month
        excess = returns - mar
        downside_diff = np.minimum(0, excess)
        downside_dev = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(12)
        if downside_dev == 0:
            return np.nan
        return np.mean(excess) / downside_dev

    # --------------------------------------------------------
    # ---------- Helper: Max Drawdown ------------------------
    def _max_drawdown(self, returns):
        """Largest peak-to-trough portfolio decline."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns.min()

    # --------------------------------------------------------
    # ---------- Helper: Average Drawdown --------------------
    def _average_drawdown(self, returns):
        """Average magnitude of all drawdowns."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns[drawdowns < 0].mean()

    # --------------------------------------------------------
    # ---------- Helper: Recovery Time -----------------------
    def _recovery_time(self, returns):
        """Number of months required to recover from max drawdown."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        min_idx = drawdowns.idxmin()

        # recovery = index where cumulative regains its peak
        recovery_idx = (cumulative[min_idx:] >= rolling_max[min_idx]).idxmax()
        return max(0, recovery_idx - min_idx)

    # --------------------------------------------------------
    # ---------- Helper: Hit Ratio ---------------------------
    def _hit_ratio(self, returns):
        """Percentage of months with positive returns."""
        return (returns > 0).mean()

    # --------------------------------------------------------
    # ---------- Helper: Beta -------------------------------
    def _beta(self, strat, bench):
        """Sensitivity of strategy to benchmark movements."""
        cov = np.cov(strat, bench)[0][1]
        var = np.var(bench)
        return cov / var

    # --------------------------------------------------------
    # ---------- Helper: Alpha ------------------------------
    def _alpha(self, strat, bench):
        """Excess return beyond what beta predicts."""
        beta = self._beta(strat, bench)
        excess_bench = bench - (self.rf_rate / 12)
        expected = (self.rf_rate / 12) + beta * excess_bench
        return np.mean(strat - expected) * 12  # annualized alpha

    # --------------------------------------------------------
    # ---------- Helper: VaR (95%) ---------------------------
    def _var(self, returns, level=0.05):
        """Value-at-Risk (VaR) at given confidence level."""
        return np.percentile(returns, 100 * level)

    # --------------------------------------------------------
    # ---------- Helper: CVaR (95%) --------------------------
    def _cvar(self, returns, level=0.05):
        """Conditional Value-at-Risk (Expected Shortfall)."""
        var = self._var(returns, level)
        return returns[returns <= var].mean()


 
    # --------------------------------------------------------
    # ---------- Helper: Recovery-Scaled Sharpe --------------
    def _recovery_scaled_sharpe(self, returns, a=0.5, b=0.02):
        """
        Recovery-Scaled Sharpe (RSS)

        A balanced risk-adjusted metric that rewards high Sharpe ratio
        but penalizes deeper drawdowns and longer recovery periods.

        RSS = Sharpe / (1 + a*|MaxDrawdown| + b*RecoveryTime)

        Parameters
        ----------
        returns : pd.Series
            Series of periodic returns (oldest → newest).
        a : float, default=0.5
            Penalty weight for maximum drawdown depth.
        b : float, default=0.02
            Penalty weight for recovery time (in months).

        Returns
        -------
        float
            Recovery-Scaled Sharpe (higher = better).
        """
        returns = returns.dropna()
        if returns.empty:
            return np.nan

        sharpe = self._sharpe_ratio(returns)
        max_dd = abs(self._max_drawdown(returns))
        recovery = self._recovery_time(returns)

        if np.isnan(sharpe) or np.isnan(max_dd) or np.isnan(recovery):
            return np.nan

        rss = sharpe / (1 + a * max_dd + b * recovery)
        return rss

    # --------------------------------------------------------
    # ---------- Core Analysis Runner ------------------------
    def _analyze(self):
        """Compute performance metrics for both strategy and benchmark."""
        s = self.df['strategy']
        b = self.df['benchmark']

        metrics = {
            "CAGR": [self._cagr(s), self._cagr(b)],
            "Volatility": [self._volatility(s), self._volatility(b)],
            "Downside Deviation": [self._downside_deviation(s), self._downside_deviation(b)],
            "Sharpe Ratio": [self._sharpe_ratio(s), self._sharpe_ratio(b)],
            "Sortino Ratio": [self._sortino_ratio(s), self._sortino_ratio(b)],
            "Max Drawdown": [self._max_drawdown(s), self._max_drawdown(b)],
            "Average Drawdown": [self._average_drawdown(s), self._average_drawdown(b)],
            "Recovery Time (Months)": [self._recovery_time(s), self._recovery_time(b)],
            "Hit Ratio": [self._hit_ratio(s), self._hit_ratio(b)],
            "Beta": [self._beta(s, b), np.nan],
            "Alpha (Annualized)": [self._alpha(s, b), np.nan],
            "VaR (95%)": [self._var(s), self._var(b)],
            "CVaR (95%)": [self._cvar(s), self._cvar(b)],
        }

        # Build a DataFrame with both columns
        df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Strategy', 'Benchmark'])
        return df

    # ================================================================
    # ===                 MONTHLY AGGREGATION LOGIC                 ===
    # ================================================================

    def monthly_avg_aggregation(self,generated_returns):
        """Aggregate average monthly strategy returns for visualization."""
        import plotly.express as px

        

        # Make a copy so we don’t modify the main df
        local_df = generated_returns.copy()
        local_df = local_df[['date','weighted_month_return']]

        # Ensure 'date' is datetime and extract numeric month (1–12)
        # Handle both datetime and Period types
        if pd.api.types.is_period_dtype(local_df['date']):
            local_df['month'] = local_df['date'].dt.to_timestamp().dt.month
        else:
            local_df['month'] = pd.to_datetime(local_df['date'], errors='coerce').dt.month


        # Group by month and calculate mean of numeric columns only
        grouped = local_df.groupby('month', as_index=False).mean(numeric_only=True)

        # Keep only month + mean strategy return
        result = grouped[['month', 'weighted_month_return']]

        return result
    

    def yearly_return_agg(self):
        df = self.df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.year
        df = df.groupby('date')[['weighted_month_return','market_return']].sum().reset_index().rename({'index':'year'})
        return df
        



    # ================================================================
    # ===                EXCEL DASHBOARD CONSTRUCTION               ===
    # ================================================================

    def full_excel_report(self, file_path: str):
        """Build a complete Excel dashboard with summary metrics and charts."""
        main_metrics_df = self._analyze()
        monthly_returns_df = self.monthly_avg_aggregation(self.df)

        workbook = xlsxwriter.Workbook(file_path, {'nan_inf_to_errors': True})

        # ------------------------------------------------------------
        # ---- Sheet 1: Main Strategy Metrics -----------------------
        sheet = workbook.add_worksheet('main strategy metrics')

        headers = ['metric type'] + main_metrics_df.columns.tolist()
        sheet.write_row(0, 0, headers)

        # Write metrics
        for row_num, (idx, row) in enumerate(main_metrics_df.iterrows(), start=1):
            sheet.write(row_num, 0, idx)
            sheet.write_row(row_num, 1, row.tolist())

        # Conditional formatting: compare strategy vs benchmark
        green_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        red_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

        # Assume strategy = column B (1), benchmark = column C (2)
        n_rows = len(main_metrics_df)

        # Green if strategy > benchmark
        sheet.conditional_format(1, 1, n_rows, 1, {
            'type': 'formula',
            'criteria': '=$B2>$C2',
            'format': green_fmt
        })

        # Red if strategy < benchmark
        sheet.conditional_format(1, 1, n_rows, 1, {
            'type': 'formula',
            'criteria': '=$B2<$C2',
            'format': red_fmt
        })

        # Apply same highlighting to benchmark column
        sheet.conditional_format(1, 2, n_rows, 2, {
            'type': 'formula',
            'criteria': '=$B2>$C2',
            'format': green_fmt
        })
        sheet.conditional_format(1, 2, n_rows, 2, {
            'type': 'formula',
            'criteria': '=$B2<$C2',
            'format': red_fmt
        })

        # ------------------------------------------------------------
        # ---- Sheet 2: Monthly Analysis ----------------------------
        month_analysis_sheet = workbook.add_worksheet('monthly analysis')
        month_analysis_sheet.write_row(0, 0, monthly_returns_df.columns.tolist())

        for row_num, row in enumerate(monthly_returns_df.itertuples(index=False), start=1):
            month_analysis_sheet.write_row(row_num, 0, row)

        # ------------------------------------------------------------
        # ---- Create and Insert Bar Chart --------------------------
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'name':       'Average Monthly Returns',
            'categories': ['monthly analysis', 1, 0, len(monthly_returns_df), 0],
            'values':     ['monthly analysis', 1, 1, len(monthly_returns_df), 1],
            'fill':       {'color': '#5A9BD5'},
            'border':     {'none': True},
        })
        chart.set_title({'name': 'Average Monthly Strategy Returns'})
        chart.set_x_axis({'name': 'Month'})
        chart.set_y_axis({'name': 'Average Return', 'major_gridlines': {'visible': False}})
        chart.set_legend({'position': 'none'})

        month_analysis_sheet.insert_chart('D2', chart, {'x_offset': 25, 'y_offset': 10})


        yearly_returns = self.yearly_return_agg()
        performence_sheet = workbook.add_worksheet('strategy historical')
        performence_sheet.write_row(0, 0, list(yearly_returns.columns))

        for row_num, row in enumerate(yearly_returns.itertuples(index=False), start=1):
            performence_sheet.write_row(row_num, 0, row)

        n_rows = len(yearly_returns) + 3


        monthly_results = self.df[['date','weighted_month_return','market_return']]
        performence_sheet.write_row(n_rows, 0, monthly_results.columns)

        at = n_rows+1

        for row_num,row in enumerate(monthly_results.itertuples(index=False),start=1):
            performence_sheet.write_row(at, 0, row)
            at = at+1

        # ============================================================
        # ---- Plot Strategy + Benchmark $100 Compounded Curves -------
        # ============================================================

        # Compute compounded values from monthly returns
        strategy_value = 100 * (1 + monthly_results['weighted_month_return']).cumprod()
        benchmark_value = 100 * (1 + monthly_results['market_return']).cumprod()

        # Append both to the DataFrame (in memory)
        monthly_results = monthly_results.copy()
        monthly_results['strategy_value'] = strategy_value
        monthly_results['benchmark_value'] = benchmark_value

        # Write these two new columns to Excel
        performence_sheet.write(n_rows, 3, 'strategy_value')
        performence_sheet.write(n_rows, 4, 'benchmark_value')
        performence_sheet.write_column(n_rows + 1, 3, monthly_results['strategy_value'])
        performence_sheet.write_column(n_rows + 1, 4, monthly_results['benchmark_value'])

        # Define key parameters
        start_row = n_rows + 1
        end_row = n_rows + len(monthly_results)
        date_col = 0
        strategy_col = 3
        benchmark_col = 4

        # Create the line chart
        equity_chart = workbook.add_chart({'type': 'line'})

        # ---- Add Strategy series ----
        equity_chart.add_series({
            'name':       'Strategy ($100 start)',
            'categories': ['strategy historical', start_row, date_col, end_row, date_col],
            'values':     ['strategy historical', start_row, strategy_col, end_row, strategy_col],
            'line':       {'color': '#4472C4', 'width': 2.25},
        })

        # ---- Add Benchmark series ----
        equity_chart.add_series({
            'name':       'Benchmark ($100 start)',
            'categories': ['strategy historical', start_row, date_col, end_row, date_col],
            'values':     ['strategy historical', start_row, benchmark_col, end_row, benchmark_col],
            'line':       {'color': '#ED7D31', 'width': 2.25},
        })

        # ---- Axis + title formatting ----
        equity_chart.set_title({'name': 'Strategy vs Benchmark Growth of $100'})
        equity_chart.set_x_axis({
            'name': 'Date',
            'date_axis': True,
            'num_format': 'yyyy-mm',
            'interval_unit': 12,       # show one tick per 12 months (approx yearly)
            'major_gridlines': {'visible': False},
        })
        equity_chart.set_y_axis({
            'name': 'Portfolio Value ($)',
            'major_gridlines': {'visible': False},
        })
        equity_chart.set_legend({'position': 'bottom'})

        # Insert chart near top of sheet
        performence_sheet.insert_chart('E2', equity_chart, {'x_offset': 25, 'y_offset': 10})

        # Close workbook after inserting chart
        workbook.close()
     
