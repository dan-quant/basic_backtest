import numpy as np
import pandas as pd
from utils import Alpha


class Alpha2(Alpha):
    '''
        mean_12(
                neg(
                    minus(const_1,div(open,close))
                )
        )
        '''

    def __init__(self, insts, dfs, start, end):
        super().__init__(insts, dfs, start, end)

    def pre_compute(self, trade_range):
        self.alphas = {}
        for inst in self.insts:
            inst_df = self.dfs[inst]
            alpha = -(1 - (inst_df.open / inst_df.close))
            alpha.replace(np.inf, 0).replace(-np.inf, 0)
            self.alphas[inst] = alpha
        return

    def post_compute(self, trade_range):
        for inst in self.insts:
            self.dfs[inst]["alpha"] = self.alphas[inst].rolling(12).mean()
            self.dfs[inst]["alpha"] = self.dfs[inst]["alpha"].fillna(method="ffill")

            self.dfs[inst]["eligible"] = self.dfs[inst]["eligible"] & (~pd.isna(self.dfs[inst]["alpha"]))

        return

    def compute_signal_distribution(self, eligibles, date):
        forecasts = {}

        for inst in eligibles:
            # get the alphas generated for each instrument on each date
            forecasts[inst] = self.dfs[inst].loc[date, "alpha"]

        absolute_scores = np.abs([score for score in forecasts.values()])
        forecast_chips = np.sum(absolute_scores)

        return forecasts, forecast_chips
