import numpy as np
import pandas as pd
from utils import get_pnl_stats
from copy import deepcopy


class Alpha1:

    def __init__(self, insts, dfs, start, end):
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.start = start
        self.end = end

    def init_portfolio_settings(self, trade_range):

        # creates the portfolio_df with the same date range as the backtest, however with datetimes as columns not index
        portfolio_df = pd.DataFrame(index=trade_range)\
            .reset_index()\
            .rename(columns={"index": "datetime"})

        # gives initial capital on the first date
        portfolio_df.at[0, "capital"] = 10000
        return portfolio_df

    def compute_meta_info(self, trade_range):
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

        # list of series to hold all the op4 series for all instruments so that we can compute the cross-sectional z-scr
        op4s = []

        for inst in self.insts:

            # cleans the self.dfs dict of dataframes (via join), so they all have the same date range as the backtest
            # computes daily returns for each instrument
            # stores eligibility of each instrument for each date based on the define criteria
            df = pd.DataFrame(index=trade_range)

            # computing the alpha as per the description above
            inst_df = self.dfs[inst]
            op1 = inst_df.volume
            op2 = (inst_df.close - inst_df.low) - (inst_df.high - inst_df.close)
            op3 = inst_df.high - inst_df.low
            op4 = op1 * op2 / op3

            # join before creating the op4 series so that the index becomes implicitly aligned
            # if we create the op4 series in the dataframe before the join, it will forward fill and backfill
            # if we create after, for the indexes (datetimes) where there was no op4 data, it will hold NaN
            self.dfs[inst] = df.join(self.dfs[inst], how="left").fillna(method="ffill").fillna(method="bfill")
            self.dfs[inst]["ret"] = self.dfs[inst]["close"].pct_change()
            self.dfs[inst]["op4"] = op4

            # we append self.dfs[inst]["op4"] and not op4 because of the index alignment. they are not the same!
            op4s.append(self.dfs[inst]["op4"])

            # sampled = pandas series of booleans that checks if the close for a given date is the same as the close
            # of the previous date
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")

            # sets eligible = 0 if the close price has been stale for 5 days in a row
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)

            # you can add any additional conditions desired in the "eligible" column, such as minimum volume, price, etc
            # this one requires the close price to change at least once every 5 days and for the close price to be > 0
            self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)

        # to compute the cross-sectional z-score (cszscre) we need to finish the loop to have all op4 series for all
        # instruments and use a temporary dataframe
        temp_df = pd.concat(op4s, axis=1)
        temp_df.columns = self.insts
        temp_df = temp_df.replace(np.inf, 0).replace(-np.inf, 0)
        cszscre_df = temp_df.fillna(method="ffill").apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)

        for inst in self.insts:
            self.dfs[inst]["alpha"] = -cszscre_df[inst].rolling(12).mean()
            self.dfs[inst]["eligible"] = self.dfs[inst]["eligible"] & (~pd.isna(self.dfs[inst]["alpha"]))
        return

    def run_simulation(self):
        print("Running backtest")

        # defines the date range for the backtest
        date_range = pd.date_range(start=self.start, end=self.end, freq="D")

        # defines which instrument were eligible to trade on each date of the date_range
        # also computes daily returns for each instrument
        self.compute_meta_info(trade_range=date_range)

        # create the portfolio_df with the initial settings
        portfolio_df = self.init_portfolio_settings(trade_range=date_range)

        for i in portfolio_df.index:
            date = portfolio_df.at[i, "datetime"]

            # filtering the instrument that were eligible for trading on the date
            eligibles = [inst for inst in self.insts if self.dfs[inst].at[date, "eligible"]]
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]

            if i != 0:
                # computes the pnl for the date
                date_prev = portfolio_df.at[i - 1, "datetime"]
                day_pnl, capital_ret = get_pnl_stats(
                    date=date,
                    prev=date_prev,
                    portfolio_df=portfolio_df,
                    insts=self.insts,
                    idx=i,
                    dfs=self.dfs
                )

            # dict to store the alpha scores for each instrument
            alpha_scores = {}

            for inst in eligibles:
                # get the alphas generated by compute_meta_info
                alpha_scores[inst] = self.dfs[inst].at[date, "alpha"]

            # sorts the alpha_scores dict by their alpha values
            alpha_scores = {k: v for k, v in sorted(alpha_scores.items(), key=lambda pair: pair[1])}

            # example of long-short strategy where we buy the top 25% and sell the bottom 25% alphas
            alpha_long = list(alpha_scores.keys())[-int(len(eligibles) / 4):]
            alpha_short = list(alpha_scores.keys())[:int(len(eligibles) / 4)]

            # compute positions and other information
            for inst in non_eligibles:
                # weights and # of units of each instrument in the portfolio, for date i, instrument inst
                portfolio_df.at[i, f"{inst} w"] = 0
                portfolio_df.at[i, f"{inst} units"] = 0

            nominal_total = 0
            for inst in eligibles:
                # forecast = position direction. 1 if long, -1 if short, 0 otherwise. No position sizing yet.
                forecast = 1 if inst in alpha_long else (-1 if inst in alpha_short else 0)
                dollar_allocation = portfolio_df.at[i, "capital"] / (len(alpha_long) + len(alpha_short))
                position = forecast * dollar_allocation / self.dfs[inst].at[date, "close"]

                # adds the units of each instrument for the date
                portfolio_df.at[i, f"{inst} units"] = position
                nominal_total += abs(position * self.dfs[inst].at[date, "close"])

            for inst in eligibles:
                units = portfolio_df.at[i, f"{inst} units"]
                nominal_inst = units * self.dfs[inst].at[date, "close"]
                inst_w = nominal_inst / nominal_total

                # adds the weight of each instrument for the date
                portfolio_df.at[i, f"{inst} w"] = inst_w

            # computes the portfolio nominal notional and leverage for the date
            portfolio_df.at[i, "nominal"] = nominal_total
            portfolio_df.at[i, "leverage"] = nominal_total / portfolio_df.at[i, "capital"]

        return portfolio_df