import numpy as np
import pandas as pd
from utils import get_pnl_stats
from copy import deepcopy


class Alpha2:

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
        portfolio_df.loc[0, "capital"] = 10000
        return portfolio_df

    def compute_meta_info(self, trade_range):
        '''
        mean_12(
                neg(
                    minus(const_1,div(open,close))
                )
        )
        '''

        for inst in self.insts:

            # cleans the self.dfs dict of dataframes (via join), so they all have the same date range as the backtest
            # computes daily returns for each instrument
            # stores eligibility of each instrument for each date based on the define criteria
            df = pd.DataFrame(index=trade_range)

            inst_df = self.dfs[inst]
            alpha = -(1 - (inst_df.open/inst_df.close))
            alpha.replace(np.inf, 0).replace(-np.inf, 0)

            self.dfs[inst] = df.join(self.dfs[inst], how="left").fillna(method="ffill").fillna(method="bfill")
            self.dfs[inst]["ret"] = self.dfs[inst]["close"].pct_change()
            self.dfs[inst]["alpha"] = alpha.rolling(12).mean()
            self.dfs[inst]["alpha"] = self.dfs[inst]["alpha"].fillna(method="ffill")

            # sampled = pandas series of booleans that checks if the close for a given date is the same as the close
            # of the previous date
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")

            # sets eligible = 0 if the close price has been stale for 5 days in a row
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)

            # you can add any additional conditions desired in the "eligible" column, such as minimum volume, price, etc
            # this one requires the close price to change at least once every 5 days and for the close price to be > 0
            self.dfs[inst]["eligible"] = eligible.astype(int)\
                & (self.dfs[inst]["close"] > 0).astype(int)\
                & (~pd.isna(self.dfs[inst]["alpha"]))

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
            date = portfolio_df.loc[i, "datetime"]

            # filtering the instrument that were eligible for trading on the date
            eligibles = [inst for inst in self.insts if self.dfs[inst].loc[date, "eligible"]]
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]

            if i != 0:
                # computes the pnl for the date
                date_prev = portfolio_df.loc[i - 1, "datetime"]
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
                # updates alphas scores using the numbers computed by compute_meta_info
                alpha_scores[inst] = self.dfs[inst].loc[date, "alpha"]

            absolute_scores = np.abs([score for score in alpha_scores.values()])
            forecast_chips = np.sum(absolute_scores)

            # compute positions and other information
            for inst in non_eligibles:
                # weights and # of units of each instrument in the portfolio, for date i, instrument inst
                portfolio_df.loc[i, f"{inst} w"] = 0
                portfolio_df.loc[i, f"{inst} units"] = 0

            nominal_total = 0
            for inst in eligibles:
                # forecast = position direction and size
                forecast = alpha_scores[inst]
                dollar_allocation = portfolio_df.loc[i, "capital"] / forecast_chips if forecast_chips !=0 else 0
                position = forecast * dollar_allocation / self.dfs[inst].loc[date, "close"]

                # adds the units of each instrument for the date
                portfolio_df.loc[i, f"{inst} units"] = position
                nominal_total += abs(position * self.dfs[inst].loc[date, "close"])

            for inst in eligibles:
                units = portfolio_df.loc[i, f"{inst} units"]
                nominal_inst = units * self.dfs[inst].loc[date, "close"]
                inst_w = nominal_inst / nominal_total

                # adds the weight of each instrument for the date
                portfolio_df.loc[i, f"{inst} w"] = inst_w

            # computes the portfolio nominal notional and leverage for the date
            portfolio_df.loc[i, "nominal"] = nominal_total
            portfolio_df.loc[i, "leverage"] = nominal_total / portfolio_df.loc[i, "capital"]

        return portfolio_df