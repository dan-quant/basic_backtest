import lzma
import random
import time
import dill as pickle
import pandas as pd
import numpy as np
from functools import wraps
from copy import deepcopy
from collections import defaultdict


def timeme(func):
    @wraps(func)
    def timediff(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"@timeme: {func.__name__} took {end - start} seconds.")
        return result
    return timediff


def load_pickle(path):
    with lzma.open(path, "rb") as file_read:
        file = pickle.load(file_read)
    return file


def save_pickle(path, obj):
    with lzma.open(path, "wb") as file_write:
        pickle.dump(obj, file_write)


# @profile
def get_pnl_stats(last_weights, last_units, close_prev, portfolio_i, ret_row, portfolio_df):
    ret_row = np.nan_to_num(ret_row, nan=0, posinf=0, neginf=0)

    day_pnl = 0
    nominal_ret = 0

    day_pnl = np.sum(last_units * close_prev * ret_row)
    nominal_ret = np.dot(last_weights, ret_row)
    capital_ret = nominal_ret * portfolio_df.at[portfolio_i - 1, "leverage"]
    portfolio_df.at[portfolio_i, "capital"] = portfolio_df.at[portfolio_i - 1, "capital"] + day_pnl
    portfolio_df.at[portfolio_i, "day_pnl"] = day_pnl
    portfolio_df.at[portfolio_i, "nominal_ret"] = nominal_ret
    portfolio_df.at[portfolio_i, "capital_ret"] = capital_ret

    return day_pnl, capital_ret


class AbstractImplementationException(Exception):
    pass


class Alpha:

    def __init__(self, insts, dfs, start, end, portfolio_vol=0.20):
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.start = start
        self.end = end
        self.portfolio_vol = portfolio_vol
        self.ewmas = [0.01] # stores the ewma of the capital returns
        self.ewstrats = [1]  # stores the ewma of the strat scalars
        self.strat_scalars = [] # stores the strat scalars

    def init_portfolio_settings(self, trade_range):

        # creates the portfolio_df with the same date range as the backtest, however with datetimes as columns not index
        portfolio_df = pd.DataFrame(index=trade_range)\
            .reset_index()\
            .rename(columns={"index": "datetime"})

        # gives initial parameters on the first date of the trade_range
        portfolio_df.at[0, "capital"] = 10000.0
        portfolio_df.at[0, "day_pnl"] = 0.0
        portfolio_df.at[0, "capital_ret"] = 0.0
        portfolio_df.at[0, "nominal_ret"] = 0.0

        return portfolio_df

    def pre_compute(self, trade_range):
        pass

    def post_compute(self, trade_range):
        pass

    def compute_signal_distribution(self, eligibles, date):
        raise AbstractImplementationException("A concrete implementation for signal generation is missing.")

    def get_strat_scalar(self, target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 253)
        return (target_vol / ann_realized_vol) * ewstrats[-1]

    # @profile
    def compute_meta_info(self, trade_range):

        self.pre_compute(trade_range=trade_range)

        vols, rets, closes, eligibles = [], [], [], []
        for inst in self.insts:
            # cleans the self.dfs dict of dataframes (via join), so they all have the same date range as the backtest
            # computes daily returns for each instrument
            # stores eligibility of each instrument for each date based on the define criteria
            df = pd.DataFrame(index=trade_range)

            # annualised rolling 30-day historical volatility. it needs to be computed before the join, otherwise we
            # will underestimate vol because the ffill and bfill in the join will result in some 0 return days (same
            # close prices). by computing the volatility before joining, the ffill and bfill will propagate the vol
            # forwards and backwards, not the close prices that will be used to compute the vol
            # this vol computation can be replaced by a more sophisticated model (GARCH EWMA, etc)
            inst_vol = self.dfs[inst]["close"].pct_change().rolling(30).std()
            self.dfs[inst] = df.join(self.dfs[inst], how="left").fillna(method="ffill").fillna(method="bfill")
            self.dfs[inst]["ret"] = self.dfs[inst]["close"].pct_change()
            self.dfs[inst]["vol"] = inst_vol
            self.dfs[inst]["vol"] = self.dfs[inst]["vol"].fillna(method="ffill").fillna(method="bfill")

            # given that the inst vol parameterizes the position size, if the vol drops too much, the position will
            # blow up to a huge number. with that, it is good practice to set a minimum vol
            self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])

            # sampled = pandas series of booleans that checks if the close for a given date is the same as the close
            # of the previous date
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")

            # sets eligible = 0 if the close price has been stale for 5 days in a row
            # we use apply(... raw=True) to avoid the creation of a new Series for each of the rolling windows
            # as the creation of Series is expensive
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x)), raw=True).fillna(0)
            self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)

            # you can add any additional conditions desired in the "eligible" column, such as minimum volume, price, etc
            # this one requires the close price to change at least once every 5 days and for the close price to be > 0
            eligibles.append(self.dfs[inst]["eligible"])
            closes.append(self.dfs[inst]["close"])
            rets.append(self.dfs[inst]["ret"])
            vols.append(self.dfs[inst]["vol"])

        # concatenate every item of the list eligibles such that each item is one column (axis=1)
        self.eligiblesdf = pd.concat(eligibles, axis=1)
        self.eligiblesdf.columns = self.insts
        self.closedf = pd.concat(closes, axis=1)
        self.closedf.columns = self.insts
        self.retdf = pd.concat(rets, axis=1)
        self.retdf.columns = self.insts
        self.voldf = pd.concat(vols, axis=1)
        self.voldf.columns = self.insts

        self.post_compute(trade_range=trade_range)

        return

    @timeme
    # @profile
    def run_simulation(self):
        date_range = pd.date_range(start=self.start, end=self.end, freq="D")
        self.compute_meta_info(trade_range=date_range)
        self.portfolio_df = self.init_portfolio_settings(trade_range=date_range)

        units_held, weights_held = [], []
        close_prev = None
        self.ewmas, self.ewstrats = [0.01], [1]
        self.strat_scalars = []

        for (data) in self.zip_data_generator():
            portfolio_i = data["portfolio_i"]
            portfolio_row = data["portfolio_row"]
            ret_i = data["ret_i"]
            ret_row = data["ret_row"]
            close_row = data["close_row"]
            eligibles_row = data["eligibles_row"]
            vol_row = data["vol_row"]

            strat_scalar = 2

            if portfolio_i != 0:
                strat_scalar = self.get_strat_scalar(
                    target_vol=self.portfolio_vol,
                    ewmas=self.ewmas,
                    ewstrats=self.ewstrats
                )
                day_pnl, capital_ret = get_pnl_stats(
                    last_weights=weights_held[-1],
                    last_units=units_held[-1],
                    close_prev=close_prev,
                    portfolio_i=portfolio_i,
                    ret_row=ret_row,
                    portfolio_df=self.portfolio_df
                )

            self.strat_scalars.append(strat_scalar)
            forecast, forecast_chips = self.compute_signal_distribution(
                eligibles_row,
                ret_i
            )

            vol_target = (self.portfolio_vol / np.sqrt(253)) * self.portfolio_df.at[portfolio_i, "capital"]
            positions = strat_scalar * \
                        forecast / forecast_chips \
                        * vol_target \
                        / (vol_row * close_row)
            positions = np.nan_to_num(positions, nan=0, posinf=0, neginf=0)
            nominal_tot = np.linalg.norm(positions * close_row, ord=1)
            units_held.append(positions)

            weights = positions * close_row / nominal_tot
            weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
            weights_held.append(weights)

            self.portfolio_df.at[portfolio_i, "nominal"] = nominal_tot
            self.portfolio_df.at[portfolio_i, "leverage"] = nominal_tot / self.portfolio_df.at[portfolio_i, "capital"]

            close_prev = close_row

        return self.portfolio_df.set_index("datetime", drop=True)

    def zip_data_generator(self):
        for (portfolio_i, portfolio_row), \
            (ret_i, ret_row), \
            (close_i, close_row), \
            (eligibles_i, eligibles_row), \
            (vol_i, vol_row) in zip(
                self.portfolio_df.iterrows(),
                self.retdf.iterrows(),
                self.closedf.iterrows(),
                self.eligiblesdf.iterrows(),
                self.voldf.iterrows()
            ):
            yield {
                "portfolio_i": portfolio_i,
                "portfolio_row": portfolio_row.values,
                "ret_i": ret_i,
                "ret_row": ret_row.values,
                "close_row": close_row.values,
                "eligibles_row": eligibles_row.values,
                "vol_row": vol_row.values
            }



class Portfolio(Alpha):
    def __init__(self, insts, dfs, start, end, strat_dfs):
        super().__init__(insts, dfs, start, end)
        self.strat_dfs = strat_dfs

    def post_compute(self, trade_range):

        # iterates over all instruments and creates 1 DataFrame for each, with the leveraged weight of the portfolio
        # under each alpha. Later in the compute signal distribution, the portfolio weights for each instrument will
        # be the sum of the weights under each alpha, scaled by the number of alphas (otherwise we would substantially
        # increase the leverage of the portfolio)

        self.positions = {}
        for inst in self.insts:
            inst_weights = pd.DataFrame(index=trade_range)
            for i in range(len(self.strat_dfs)):
                inst_weights[i] = self.strat_dfs[i][f"{inst} w"] * self.strat_dfs[i]["leverage"]
                inst_weights = inst_weights.fillna(method="ffill").fillna(0.0)
            self.positions[inst] = inst_weights

    def compute_signal_distribution(self, eligibles, date):
        # we use a default dict so that we dont raise an exception in the loop below whne trying to add to something undefined
        forecasts = defaultdict(float)

        for inst in self.insts:
            for i in range(len(self.strat_dfs)):

                # the below works because forecasts is a defaultdict(float)
                forecasts[inst] += self.positions[inst].at[date, i] * (1/len(self.strat_dfs))

        return forecasts, np.sum(np.abs(list(forecasts.values())))
