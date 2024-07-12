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

    # @profile
    def post_compute(self, trade_range):
        temp = []
        for inst in self.insts:
            temp.append(self.dfs[inst]["alpha"])

        self.alphadf = pd.concat(temp, axis=1)
        self.alphadf.columns = self.insts
        self.alphadf = self.alphadf.fillna(method="ffill")
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(self.alphadf)).astype(np.int32)

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.alphadf.loc[date].values
        return forecasts
