import numpy as np
import pandas as pd
from utils import Alpha


class Alpha1(Alpha):
    '''
    https://hangukquant.substack.com/p/formulaic-alphas-5ae
    mean_12(
        neg(
            cszscre(
                mult(
                    volume,
                    div(
                        minus(minus(close,low),minus(high,close)),
                        minus(high,low)
                    )
                )
            )
        )
    )

    cszscre = cross-sectional z-score
    '''

    def __init__(self, insts, dfs, start, end):
        super().__init__(insts, dfs, start, end)

    def pre_compute(self, trade_range):
        self.op4s = {}

        for inst in self.insts:
            # computing the alpha as per the description above
            inst_df = self.dfs[inst]
            op1 = inst_df.volume
            op2 = (inst_df.close - inst_df.low) - (inst_df.high - inst_df.close)
            op3 = inst_df.high - inst_df.low
            op4 = op1 * op2 / op3
            self.op4s[inst] = op4

        return

    # @profile
    def post_compute(self, trade_range):
        temp = []
        for inst in self.insts:
            self.dfs[inst]["op4"] = self.op4s[inst]
            temp.append(self.dfs[inst]["op4"])

        temp_df = pd.concat(temp, axis=1)
        temp_df.columns = self.insts

        # changed the default value from 0 to np.nan as 0 affects the cszscre computation, while np.nan gets ignored
        temp_df = temp_df.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        # we cant use a numpy array below (raw=True) because numpy is NOT NaN aware, so if there is one NaN in the array
        # all the results of the computation will be NaN
        # on the other hand, pandas series ARE NaN aware, so the computation just ignores NaN values and also preserves
        # the location of the NaN in the series
        # HOWEVER, Numpy has np.nanmean() and np.nanstd(), which are NaN aware version of the functions

        cszscre_df = temp_df.fillna(method="ffill").apply(lambda x: (x - np.nanmean(x)) / np.nanstd(x), axis=1, raw=True)

        alphas = []
        for inst in self.insts:
            self.dfs[inst]["alpha"] = -cszscre_df[inst].rolling(12).mean()
            alphas.append(self.dfs[inst]["alpha"])

        alphadf = pd.concat(alphas, axis=1)
        alphadf.columns = self.insts
        self.alphadf = alphadf
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf)).astype(np.int32)
        masked_df = self.alphadf / self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)
        num_eligibles = self.eligiblesdf.sum(axis=1)
        rankdf = masked_df.rank(axis=1, method="average", na_option="keep", ascending=True)
        shortdf = rankdf.apply(lambda col: col <= num_eligibles.values/4, axis=0, raw=True)
        longdf = rankdf.apply(lambda col: col > np.ceil(num_eligibles.values - num_eligibles.values / 4), axis=0, raw=True)
        forecast_df = -shortdf.astype(np.int32) + longdf.astype(np.int32)
        self.forecast_df = forecast_df
        return

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts
