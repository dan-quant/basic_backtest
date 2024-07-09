import pytz
import yfinance
import threading
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from utils import load_pickle, save_pickle, Portfolio, timeme
from alpha1 import Alpha1
from alpha2 import Alpha2
from alpha3 import Alpha3
import matplotlib.pyplot as plt



TIMEZONE = pytz.utc
PERIOD_START_DATE = datetime(2010, 1, 1, tzinfo=TIMEZONE)
PERIOD_END_DATE = datetime.now(tz=TIMEZONE)
NUMBER_OF_TICKERS = 20


def get_sp500_tickers():

    # gets all the tickers of the S&P 500

    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content, 'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))
    tickers = list(df[0].Symbol)
    return tickers


def get_history(ticker, period_start, period_end, granularity="1d", tries=0):

    # tries at least 5 times to get the data history for the given ticker

    try:
        df = yfinance.Ticker(ticker).history(
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True
        ).reset_index()

    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)

        return pd.DataFrame()

    df = df.rename(columns={
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    if df.empty:
        return pd.DataFrame()

    df["datetime"] = df["datetime"].dt.tz_convert(TIMEZONE).dt.floor("D")
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df = df.set_index("datetime", drop=True)

    return df


def get_histories(tickers, period_starts, period_ends, granularity="1d"):

    # multithreaded function that gets the data for each ticker and returns the tickers and dataframes with the data

    dfs = [None]*len(tickers)

    def _helper(i):
        print(tickers[i])
        df = get_history(tickers[i], period_starts[i], period_ends[i], granularity)
        dfs[i] = df

    threads = [threading.Thread(target=_helper, args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]

    return tickers, dfs


def get_ticker_dfs(start, end):
    # checks if there is a file saved with the data, otherwise requests it from yfinance and saves the data in a file
    try:
        tickers, ticker_dfs = load_pickle("dataset.obj")

    except Exception as err:

        # gets all S&P 500 tickers
        tickers = get_sp500_tickers()
        starts = [start]*len(tickers)
        ends = [end] * len(tickers)


        tickers, dfs = get_histories(tickers, starts, ends, granularity="1d")
        ticker_dfs = {ticker: df for ticker, df in zip(tickers, dfs)}
        save_pickle("dataset.obj", (tickers, ticker_dfs))

    return tickers, ticker_dfs


def plot_vol(r):
    vol = r.rolling(25).std() * np.sqrt(253)
    plt.plot(vol)
    plt.show()
    plt.close()


def main():
    tickers, ticker_dfs = get_ticker_dfs(start=PERIOD_START_DATE,
                                         end=PERIOD_END_DATE)

    alpha1 = Alpha1(insts=tickers[:NUMBER_OF_TICKERS],
                    dfs=ticker_dfs,
                    start=PERIOD_START_DATE,
                    end=PERIOD_END_DATE
                    )

    # alpha2 = Alpha2(insts=tickers[:NUMBER_OF_TICKERS],
    #                 dfs=ticker_dfs,
    #                 start=PERIOD_START_DATE,
    #                 end=PERIOD_END_DATE
    #                 )
    #
    # alpha3 = Alpha3(insts=tickers[:NUMBER_OF_TICKERS],
    #                 dfs=ticker_dfs,
    #                 start=PERIOD_START_DATE,
    #                 end=PERIOD_END_DATE
    #                 )

    df1 = alpha1.run_simulation()
    # df2 = alpha2.run_simulation()
    # df3 = alpha3.run_simulation()

    print(f"df1: {list(df1.capital)[-1]}")
    # print(f"df2: {list(df2.capital)[-1]}")
    # print(f"df3: {list(df3.capital)[-1]}")

    # portfolio = Portfolio(insts=tickers[:10],
    #                       dfs=ticker_dfs,
    #                       start=PERIOD_START_DATE,
    #                       end=PERIOD_END_DATE,
    #                       strat_dfs=[df1, df2, df3]
    #                       )
    #
    # portfolio_df = portfolio.run_simulation()


if __name__ == '__main__':
    main()
'''
### Benchmark for 20 tickers:

@timeme: run_simulation took 49.95934462547302 seconds.
@timeme: run_simulation took 52.4849579334259 seconds.
@timeme: run_simulation took 52.523226261138916 seconds.
df1: 31346.569762350704
df1: 9731.451568466826
df1: 110581.37703868462

Sun Jun 30 23:14:04 2024    stats.txt

         543680051 function calls (540598892 primitive calls) in 291.647 seconds

   Ordered by: cumulative time
   List reduced from 5760 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    819/1    0.007    0.000  291.648  291.648 {built-in method builtins.exec}
        1    0.056    0.056  291.648  291.648 main.py:1(<module>)
        1    0.001    0.001  290.118  290.118 main.py:119(main)
        3    0.059    0.020  287.082   95.694 utils.py:13(timediff)
        3    4.036    1.345  287.022   95.674 utils.py:142(run_simulation)
  3741219   12.550    0.000  153.360    0.000 indexing.py:1089(__getitem__)
   730710    3.634    0.000   93.153    0.000 indexing.py:831(__setitem__)
  3423759    7.559    0.000   76.254    0.000 frame.py:3847(_get_value)
730848/730710    4.055    0.000   66.827    0.000 indexing.py:1689(_setitem_with_indexer)
    15882    1.795    0.000   63.931    0.004 utils.py:34(get_pnl_stats)
   730710    2.690    0.000   54.910    0.000 indexing.py:1839(_setitem_with_indexer_split_path)
        3    0.006    0.002   52.745   17.582 utils.py:97(compute_meta_info)
      280    0.001    0.000   48.637    0.174 rolling.py:558(_apply)
      280    0.001    0.000   48.636    0.174 rolling.py:456(_apply_blockwise)
      280    0.002    0.000   48.634    0.174 rolling.py:436(_apply_series)
      280    0.001    0.000   48.607    0.174 rolling.py:591(homogeneous_func)
      280    0.012    0.000   48.602    0.174 rolling.py:597(calc)
       60    0.000    0.000   48.577    0.810 rolling.py:1892(apply)
       60    0.000    0.000   48.576    0.810 rolling.py:1353(apply)
       60    1.086    0.018   48.549    0.809 rolling.py:1413(apply_func)

    ### Replacing .loc by .at: 33% improvement
    
@timeme: run_simulation took 33.58416390419006 seconds.
@timeme: run_simulation took 35.01650428771973 seconds.
@timeme: run_simulation took 34.62669062614441 seconds.
df1: 31346.569762350704
df2: 9731.451568466826
df3: 110581.37703868462

Wed Jul  3 20:59:49 2024    stats_with_at.txt

         273184361 function calls (273025451 primitive calls) in 182.729 seconds

   Ordered by: cumulative time
   List reduced from 5768 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    819/1    0.009    0.000  182.730  182.730 {built-in method builtins.exec}
        1    0.057    0.057  182.728  182.728 main.py:1(<module>)
        1    0.001    0.001  177.659  177.659 main.py:119(main)
        3    0.063    0.021  174.621   58.207 utils.py:13(timediff)
        3    3.299    1.100  174.558   58.186 utils.py:142(run_simulation)
  3424893    4.848    0.000   87.720    0.000 indexing.py:2412(__getitem__)
  3424893    4.397    0.000   79.145    0.000 indexing.py:2362(__getitem__)
  3424893    7.249    0.000   73.696    0.000 frame.py:3847(_get_value)
        3    0.006    0.002   53.169   17.723 utils.py:97(compute_meta_info)
      280    0.001    0.000   49.187    0.176 rolling.py:558(_apply)
      280    0.001    0.000   49.186    0.176 rolling.py:456(_apply_blockwise)
      280    0.002    0.000   49.184    0.176 rolling.py:436(_apply_series)
      280    0.002    0.000   49.153    0.176 rolling.py:591(homogeneous_func)
      280    0.013    0.000   49.148    0.176 rolling.py:597(calc)
       60    0.000    0.000   49.109    0.818 rolling.py:1892(apply)
       60    0.000    0.000   49.109    0.818 rolling.py:1353(apply)
       60    1.144    0.019   49.080    0.818 rolling.py:1413(apply_func)
    15891    1.566    0.000   41.964    0.003 utils.py:34(get_pnl_stats)
  3425859    3.826    0.000   31.419    0.000 frame.py:4243(_get_item_cache)
  2487012    6.622    0.000   29.015    0.000 datetimes.py:536(get_loc)

    ### Line profiling results

Wrote profile results to main.py.lprof
Timer unit: 1e-06 s

Total time: 101.2 s
Function: compute_meta_info at line 97

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    97                                               @profile
    98                                               def compute_meta_info(self, trade_range):
    99
   100         3     151699.7  50566.6      0.1          self.pre_compute(trade_range=trade_range)
   101
   102        63         53.8      0.9      0.0          for inst in self.insts:
   103
   104                                                       # cleans the self.dfs dict of dataframes (via join), so they all have the same date range as the backtest
   105                                                       # computes daily returns for each instrument
   106                                                       # stores eligibility of each instrument for each date based on the define criteria
   107        60      44187.5    736.5      0.0              df = pd.DataFrame(index=trade_range)
   108
   109                                                       # annualised rolling 30-day historical volatility. it needs to be computed before the join, otherwise we
   110                                                       # will underestimate vol because the ffill and bfill in the join will result in some 0 return days (same
   111                                                       # close prices). by computing the volatility before joining, the ffill and bfill will propagate the vol
   112                                                       # forwards and backwards, not the close prices that will be used to compute the vol
   113                                                       # this vol computation can be replaced by a more sophisticated model (GARCH EWMA, etc)
   114        60      97557.2   1626.0      0.1              inst_vol = self.dfs[inst]["close"].pct_change().rolling(30).std()
   115        60     164346.1   2739.1      0.2              self.dfs[inst] = df.join(self.dfs[inst], how="left").fillna(method="ffill").fillna(method="bfill")
   116        60      97192.6   1619.9      0.1              self.dfs[inst]["ret"] = self.dfs[inst]["close"].pct_change()
   117        60      86302.2   1438.4      0.1              self.dfs[inst]["vol"] = inst_vol
   118        60      43498.8    725.0      0.0              self.dfs[inst]["vol"] = self.dfs[inst]["vol"].fillna(method="ffill").fillna(method="bfill")
   119
   120                                                       # given that the inst vol parameterizes the position size, if the vol drops too much, the position will
   121                                                       # blow up to a huge number. with that, it is good practice to set a minimum vol
   122        60      34455.1    574.3      0.0              self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
   123
   124                                                       # sampled = pandas series of booleans that checks if the close for a given date is the same as the close
   125                                                       # of the previous date
   126        60      48888.7    814.8      0.0              sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")
   127
   128                                                       # sets eligible = 0 if the close price has been stale for 5 days in a row  
   129        60   93561894.0    2e+06     92.5              eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)    
   130
   131                                                       # you can add any additional conditions desired in the "eligible" column, such as minimum volume, price, etc
   132                                                       # this one requires the close price to change at least once every 5 days and for the close price to be > 0
   133        60      97335.7   1622.3      0.1              self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)
   134
   135         3    6772300.0    2e+06      6.7          self.post_compute(trade_range=trade_range)
   136
   137         3          2.1      0.7      0.0          return


    ### changed from: eligible = sampled.rolling(5).apply(lambda x: int(np.any(x)), raw=False).fillna(0)
    to: eligible = sampled.rolling(5).apply(lambda x: int(np.any(x)), raw=True).fillna(0)
    30% improvement
    
@timeme: run_simulation took 23.91638159751892 seconds.
@timeme: run_simulation took 25.385700941085815 seconds.
@timeme: run_simulation took 24.570518493652344 seconds.
df1: 31346.569762350704
df2: 9731.451568466826
df3: 110581.37703868462
    
    Wrote profile results to main.py.lprof
Timer unit: 1e-06 s

Total time: 11.3655 s
Function: compute_meta_info at line 97

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    97                                               @profile
    98                                               def compute_meta_info(self, trade_range):
    99
   100         3     147947.9  49316.0      1.3          self.pre_compute(trade_range=trade_range)
   101
   102        63         54.9      0.9      0.0          for inst in self.insts:
   103
   104                                                       # cleans the self.dfs dict of dataframes (via join), so they all have the same date range as the backtest
   105                                                       # computes daily returns for each instrument
   106                                                       # stores eligibility of each instrument for each date based on the define criteria
   107        60      43482.6    724.7      0.4              df = pd.DataFrame(index=trade_range)
   108
   109                                                       # annualised rolling 30-day historical volatility. it needs to be computed before the join, otherwise we
   110                                                       # will underestimate vol because the ffill and bfill in the join will result in some 0 return days (same
   111                                                       # close prices). by computing the volatility before joining, the ffill and bfill will propagate the vol
   112                                                       # forwards and backwards, not the close prices that will be used to compute the vol
   113                                                       # this vol computation can be replaced by a more sophisticated model (GARCH EWMA, etc)
   114        60      95204.6   1586.7      0.8              inst_vol = self.dfs[inst]["close"].pct_change().rolling(30).std()
   115        60     161614.4   2693.6      1.4              self.dfs[inst] = df.join(self.dfs[inst], how="left").fillna(method="ffill").fillna(method="bfill")
   116        60      93526.9   1558.8      0.8              self.dfs[inst]["ret"] = self.dfs[inst]["close"].pct_change()
   117        60      83033.6   1383.9      0.7              self.dfs[inst]["vol"] = inst_vol
   118        60      41325.3    688.8      0.4              self.dfs[inst]["vol"] = self.dfs[inst]["vol"].fillna(method="ffill").fillna(method="bfill")
   119
   120                                                       # given that the inst vol parameterizes the position size, if the vol drops too much, the position will
   121                                                       # blow up to a huge number. with that, it is good practice to set a minimum vol
   122        60      32693.9    544.9      0.3              self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
   123
   124                                                       # sampled = pandas series of booleans that checks if the close for a given date is the same as the close
   125                                                       # of the previous date
   126        60      47209.4    786.8      0.4              sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")
   127
   128                                                       # sets eligible = 0 if the close price has been stale for 5 days in a row  
   129                                                       # we use apply(... raw=True) to avoid the creation of a new Series for each of the rolling windows
   130                                                       # as the creation of Series is expensive
   131        60    4345710.7  72428.5     38.2              eligible = sampled.rolling(5).apply(lambda x: int(np.any(x)), raw=True).fillna(0)
   132
   133                                                       # you can add any additional conditions desired in the "eligible" column, such as minimum volume, price, etc
   134                                                       # this one requires the close price to change at least once every 5 days and for the close price to be > 0
   135        60      94087.7   1568.1      0.8              self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)
   136
   137         3    6179577.2    2e+06     54.4          self.post_compute(trade_range=trade_range)
   138
   139         3          2.0      0.7      0.0          return

    ### line profiling utils.run_simulation and alpha1.post_compute
    
    Wrote profile results to main.py.lprof
Timer unit: 1e-06 s

Total time: 6.55374 s
Function: post_compute at line 43

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    43                                               @profile
    44                                               def post_compute(self, trade_range):
    45         1          0.6      0.6      0.0          temp = []
    46        21         14.5      0.7      0.0          for inst in self.insts:
    47        20      28497.8   1424.9      0.4              self.dfs[inst]["op4"] = self.op4s[inst]
    48        20       2197.6    109.9      0.0              temp.append(self.dfs[inst]["op4"])
    49
    50         1       7898.2   7898.2      0.1          temp_df = pd.concat(temp, axis=1)
    51         1        148.7    148.7      0.0          temp_df.columns = self.insts
    52         1       1166.1   1166.1      0.0          temp_df = temp_df.replace(np.inf, 0).replace(-np.inf, 0)
    53         1    6470301.9    6e+06     98.7          cszscre_df = temp_df.fillna(method="ffill").apply(lambda x: (x - np.mean(x)) / np.std(x), axis=1)
    54
    55        21         15.8      0.8      0.0          for inst in self.insts:
    56        20      25643.7   1282.2      0.4              self.dfs[inst]["alpha"] = -cszscre_df[inst].rolling(12).mean()
    57        20      17850.2    892.5      0.3              self.dfs[inst]["eligible"] = self.dfs[inst]["eligible"] & (~pd.isna(self.dfs[inst]["alpha"]))

Total time: 234.524 s
Function: run_simulation at line 145

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   145                                               @timeme
   146                                               @profile
   147                                               def run_simulation(self):
   148                                                   # defines the date range for the backtest
   149         3       1788.2    596.1      0.0          date_range = pd.date_range(start=self.start, end=self.end, freq="D")
   150
   151                                                   # defines which instrument were eligible to trade on each date of the date_range
   152                                                   # also computes daily returns for each instrument
   153         3   11853261.7    4e+06      5.1          self.compute_meta_info(trade_range=date_range)
   154
   155                                                   # create the portfolio_df with the initial settings
   156         3      15620.7   5206.9      0.0          portfolio_df = self.init_portfolio_settings(trade_range=date_range)
   157
   158     15897      18979.4      1.2      0.0          for i in portfolio_df.index:
   159     15894    1969246.6    123.9      0.8              date = portfolio_df.at[i, "datetime"]
   160
   161                                                       # filtering the instrument that were eligible for trading on the date      
   162     15894   12194266.8    767.2      5.2              eligibles = [inst for inst in self.insts if self.dfs[inst].at[date, "eligible"]]
   163     15894     154559.7      9.7      0.1              non_eligibles = [inst for inst in self.insts if inst not in eligibles]     
   164
   165                                                       # initial scalar applied to forecasts, used for volatility targeting       
   166     15894       6637.6      0.4      0.0              strat_scalar = 2
   167
   168     15894       8027.5      0.5      0.0              if i != 0:
   169                                                           # computes the pnl for the date
   170     15891     620825.1     39.1      0.3                  date_prev = portfolio_df.at[i - 1, "datetime"]
   171
   172                                                           # gets the scalar used for strategy vol targeting
   173     31782     126313.9      4.0      0.1                  strat_scalar = self.get_strat_scalar(
   174     15891       8050.3      0.5      0.0                      target_vol=self.portfolio_vol,
   175     15891       6720.7      0.4      0.0                      ewmas=self.ewmas,
   176     15891       6978.0      0.4      0.0                      ewstrats=self.ewstrats
   177                                                           )
   178
   179     31782   77516457.0   2439.0     33.1                  day_pnl, capital_ret = get_pnl_stats(
   180     15891       5918.1      0.4      0.0                      date=date,
   181     15891       5776.1      0.4      0.0                      prev=date_prev,
   182     15891       5624.8      0.4      0.0                      portfolio_df=portfolio_df,
   183     15891       6483.0      0.4      0.0                      insts=self.insts,
   184     15891       5713.4      0.4      0.0                      idx=i,
   185     15891       6088.8      0.4      0.0                      dfs=self.dfs
   186                                                           )
   187
   188                                                           # updating ewma of capital returns and appending to the list
   189     31782      24876.5      0.8      0.0                  self.ewmas.append(
   190     15891      35343.7      2.2      0.0                      0.06 * (capital_ret ** 2) + 0.94 * self.ewmas[-1] if capital_ret != 0 else self.ewmas[-1])
   191
   192
   193                                                           # updating ewma of strat scalars and appending to the list
   194     31782      17614.5      0.6      0.0                  self.ewstrats.append(
   195     15891      16690.3      1.1      0.0                      0.06 * strat_scalar + 0.94 * self.ewstrats[-1] if capital_ret != 0 else self.ewstrats[-1])
   196
   197     15894      12130.6      0.8      0.0              self.strat_scalars.append(strat_scalar)
   198
   199                                                       # generates the signal for all instruments on this date
   200     15894   12342755.9    776.6      5.3              forecasts, forecast_chips = self.compute_signal_distribution(eligibles, date)
   201
   202                                                       # compute positions and other information
   203     36267      20131.7      0.6      0.0              for inst in non_eligibles:
   204                                                           # weights and # of units of each instrument in the portfolio, for date i, instrument inst
   205     20373    1351411.1     66.3      0.6                  portfolio_df.at[i, f"{inst} w"] = 0
   206     20373    1276468.5     62.7      0.5                  portfolio_df.at[i, f"{inst} units"] = 0
   207
   208                                                       # position size adjustment for volatility targeting
   209                                                       # we convert the annualised vol target to a daily vol expressed in dollar terms
   210     15894    1413365.5     88.9      0.6              vol_target = (self.portfolio_vol / np.sqrt(253)) * portfolio_df.at[i, "capital"]
   211
   212     15894       8673.3      0.5      0.0              nominal_total = 0
   213    313401     166106.6      0.5      0.1              for inst in eligibles:
   214                                                           # forecast is the position direction and size
   215    297507     160460.6      0.5      0.1                  forecast = forecasts[inst]
   216
   217                                                           # scaled_forecast = % of inst alpha / sum of absolute alpha of all insts
   218    297507     295301.5      1.0      0.1                  scaled_forecast = forecast / forecast_chips if forecast_chips != 0 else 0
   219    297507     175990.4      0.6      0.1                  inst_vol_target = scaled_forecast * vol_target
   220    297507   22850346.8     76.8      9.7                  inst_vol = self.dfs[inst].at[date, "vol"] * self.dfs[inst].at[date, "close"]
   221    297507     221702.6      0.7      0.1                  position = strat_scalar * inst_vol_target / inst_vol
   222
   223                                                           # adds the units of each instrument for the date
   224    297507   18398991.4     61.8      7.8                  portfolio_df.at[i, f"{inst} units"] = position
   225    297507   12425736.8     41.8      5.3                  nominal_total += abs(position * self.dfs[inst].at[date, "close"])      
   226
   227    313401     168824.3      0.5      0.1              for inst in eligibles:
   228    297507   24068615.5     80.9     10.3                  units = portfolio_df.at[i, f"{inst} units"]
   229    297507   12454066.4     41.9      5.3                  nominal_inst = units * self.dfs[inst].at[date, "close"]
   230    297507     203223.9      0.7      0.1                  inst_w = nominal_inst / nominal_total
   231
   232                                                           # adds the weight of each instrument for the date
   233    297507   18666842.5     62.7      8.0                  portfolio_df.at[i, f"{inst} w"] = inst_w
   234
   235                                                       # computes the portfolio nominal notional and leverage for the date        
   236     15894     954168.1     60.0      0.4              portfolio_df.at[i, "nominal"] = nominal_total
   237     15894    2239251.6    140.9      1.0              portfolio_df.at[i, "leverage"] = nominal_total / portfolio_df.at[i, "capital"]
   238
   239         3      11625.8   3875.3      0.0          return portfolio_df.set_index("datetime", drop=True)




'''