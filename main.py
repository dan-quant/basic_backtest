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

    alpha2 = Alpha2(insts=tickers[:NUMBER_OF_TICKERS],
                    dfs=ticker_dfs,
                    start=PERIOD_START_DATE,
                    end=PERIOD_END_DATE
                    )

    alpha3 = Alpha3(insts=tickers[:NUMBER_OF_TICKERS],
                    dfs=ticker_dfs,
                    start=PERIOD_START_DATE,
                    end=PERIOD_END_DATE
                    )

    df1 = alpha1.run_simulation()
    df2 = alpha2.run_simulation()
    df3 = alpha3.run_simulation()

    print(f"df1: {list(df1.capital)[-1]}")
    print(f"df2: {list(df2.capital)[-1]}")
    print(f"df3: {list(df3.capital)[-1]}")

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
benchmark for 20 tickers:
@timeme: run_simulation took 49.95934462547302 seconds.
@timeme: run_simulation took 52.4849579334259 seconds.
@timeme: run_simulation took 52.523226261138916 seconds.
df1: 31346.569762350704
df1: 9731.451568466826
df1: 110581.37703868462

replacing .loc by .at: 33% improvement
@timeme: run_simulation took 33.58416390419006 seconds.
@timeme: run_simulation took 35.01650428771973 seconds.
@timeme: run_simulation took 34.62669062614441 seconds.
df1: 31346.569762350704
df2: 9731.451568466826
df3: 110581.37703868462
'''