import numpy as np
import pandas as pd
import math

class MonthlyWeightMaker:
    def __init__(self, avg_returns_table):
        self.avg_returns_table = avg_returns_table

    def mcq_alpha_weights(self,
                                mean_returns=None,
                                P_avg=0.20,
                                k=0.5,
                                P_min=0.10,
                                P_max=0.40,
                                month_equal_weight=0.20):
            mean_return_df = self.avg_returns_table
        
            """
            Apply exponential tilt weighting to month-of-year average returns.

            Parameters
            ----------
            mean_return_df : pd.DataFrame, optional
                Must have columns ['month', 'weighted_month_return'].
            mean_returns : list or np.array, optional
                12 monthly mean returns if you don't provide a DataFrame.
            P_avg : float
                Target average buy fraction across the year (e.g., 0.20 for 20%).
            k : float
                Aggressiveness of the exponential tilt.
            P_min, P_max : float
                Minimum and maximum buy percentages (fractions, not percents).
            month_equal_weight : float
                Equal weighting baseline for all months (same as P_avg by default).

            Returns
            -------
            dict : JSON-style dictionary of month weights (as rounded whole percentages)
            """

            # --- 1) Prepare the data ---
            if mean_return_df is not None:
                df = mean_return_df.copy()
            elif mean_returns is not None:
                df = pd.DataFrame({
                    'month': range(1, 13),
                    'weighted_month_return': mean_returns
                })
            else:
                raise ValueError("Provide either mean_return_df or mean_returns.")

            # --- 2) Compute mean & std ---
            mean_ret = df['weighted_month_return'].mean()
            std_ret = df['weighted_month_return'].std(ddof=0)

            # --- 3) Compute z-scores ---
            df['z'] = (df['weighted_month_return'] - mean_ret) / (std_ret + 1e-12)

            # --- 4) Exponential tilt ---
            df['score'] = np.exp(k * df['z'])
            df['adj_score'] = df['score'] / df['score'].mean()

            # --- 5) Scale to baseline equal weight ---
            df['raw_buy'] = month_equal_weight * df['adj_score']

            # --- 6) Clip and renormalize to maintain average P_avg ---
            df['clipped'] = df['raw_buy'].clip(P_min, P_max)
            scale = P_avg / df['clipped'].mean()
            df['final'] = df['clipped'] * scale

            # --- 7) Convert to percentages (whole numbers) ---
            df['buy_percent'] = (df['final'] * 100).round(0).astype(int)

            # --- 8) Build dictionary ---
            weights_dict = {
                "": "this json file takes in the numerical encodings for each month and "
                    "takes the top x% of stocks with the lowest composite rankings to be bought"
            }
            for _, row in df.iterrows():
                weights_dict[f"{int(row['month']):02d}"] = str(row['buy_percent'])

            return weights_dict
