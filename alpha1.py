import numpy as np
import pandas as pd
# from utils import Alpha
from utils import EfficientAlpha as Alpha

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
        temp_df = temp_df.replace(np.inf, 0).replace(-np.inf, 0)
        cszscre_df = temp_df.fillna(method="ffill").apply(lambda x: (x - np.mean(x)) / np.std(x), axis=1)

        for inst in self.insts:
            self.dfs[inst]["alpha"] = -cszscre_df[inst].rolling(12).mean()
            self.dfs[inst]["eligible"] = self.dfs[inst]["eligible"] & (~pd.isna(self.dfs[inst]["alpha"]))

    def compute_signal_distribution(self, eligibles, date):
        alpha_scores = {}

        for inst in eligibles:
            # get the alphas generated for each instrument on each date
            alpha_scores[inst] = self.dfs[inst].at[date, "alpha"]

        # sorts the alpha_scores dict by their alpha values
        alpha_scores = {k: v for k, v in sorted(alpha_scores.items(), key=lambda pair: pair[1])}

        # example of long-short strategy where we buy the top 25% and sell the bottom 25% alphas
        alpha_long = list(alpha_scores.keys())[-int(len(eligibles) / 4):]
        alpha_short = list(alpha_scores.keys())[:int(len(eligibles) / 4)]

        forecasts = {}
        for inst in eligibles:
            forecasts[inst] = 1 if inst in alpha_long else (-1 if inst in alpha_short else 0)

        forecast_chips = np.sum(np.abs(list(forecasts.values())))

        return forecasts, forecast_chips
