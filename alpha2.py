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

    # @profile
    def post_compute(self, trade_range):
        temp = []
        for inst in self.insts:
            self.dfs[inst]["alpha"] = self.alphas[inst].rolling(12).mean()
            temp.append(self.dfs[inst]["alpha"])

        self.alphadf = pd.concat(temp, axis=1)
        self.alphadf.columns = self.insts
        self.alphadf = self.alphadf.fillna(method="ffill")
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(self.alphadf)).astype(np.int32)

        return

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.alphadf.loc[date].values
        return forecasts
