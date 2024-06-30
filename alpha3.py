import numpy as np
import pandas as pd
from utils import Alpha


class Alpha3(Alpha):
    '''
        ma_faster > ma_slower ? buy : 0 (this is a long-only signal)
        1. fast_crossover
        2. medium_crossover
        3. slow_crossover
        plus(
            mean_10(close) > mean_50(close),
            mean_20(close) > mean_100(close),
            mean_50(close) > mean_200(close),
        )

        plus can be = 0, 1, 2, 3
        '''

    def __init__(self, insts, dfs, start, end):
        super().__init__(insts, dfs, start, end)

    def pre_compute(self, trade_range):
        for inst in self.insts:
            # computing the alpha as per the description above
            inst_df = self.dfs[inst]
            fast = np.where(inst_df["close"].rolling(10).mean() > inst_df["close"].rolling(50).mean(), 1, 0)
            medium = np.where(inst_df["close"].rolling(20).mean() > inst_df["close"].rolling(100).mean(), 1, 0)
            slow = np.where(inst_df["close"].rolling(50).mean() > inst_df["close"].rolling(200).mean(), 1, 0)
            alpha = fast + medium + slow
            self.dfs[inst]["alpha"] = alpha
        return

    def post_compute(self, trade_range):
        for inst in self.insts:
            self.dfs[inst]["eligible"] = self.dfs[inst]["eligible"] & (~pd.isna(self.dfs[inst]["alpha"]))

    def compute_signal_distribution(self, eligibles, date):
        forecasts = {}

        for inst in eligibles:
            # get the alphas generated for each instrument on each date
            forecasts[inst] = self.dfs[inst].loc[date, "alpha"]

        absolute_scores = np.abs([score for score in forecasts.values()])
        forecast_chips = np.sum(absolute_scores)

        return forecasts, forecast_chips
