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


@timeme
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


@timeme
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


@timeme
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


if __name__ == '__main__':
    tickers, ticker_dfs = get_ticker_dfs(start=PERIOD_START_DATE,
                                         end=PERIOD_END_DATE)

    alpha1 = Alpha1(insts=tickers[:10],
                    dfs=ticker_dfs,
                    start=PERIOD_START_DATE,
                    end=PERIOD_END_DATE
                    )

    alpha2 = Alpha2(insts=tickers[:10],
                    dfs=ticker_dfs,
                    start=PERIOD_START_DATE,
                    end=PERIOD_END_DATE
                    )

    alpha3 = Alpha3(insts=tickers[:10],
                    dfs=ticker_dfs,
                    start=PERIOD_START_DATE,
                    end=PERIOD_END_DATE
                    )

    # df1 = alpha1.run_simulation()
    # df2 = alpha2.run_simulation()
    # df3 = alpha3.run_simulation()

    df1, df2, df3 = load_pickle("simulations_strat_vol_targeting.obj")
    df1 = df1.set_index("datetime", drop=True)
    df2 = df2.set_index("datetime", drop=True)
    df3 = df3.set_index("datetime", drop=True)

    portfolio = Portfolio(insts=tickers[:10],
                          dfs=ticker_dfs,
                          start=PERIOD_START_DATE,
                          end=PERIOD_END_DATE,
                          strat_dfs=[df1, df2, df3]
                          )

    # portfolio_df = portfolio.run_simulation()

    portfolio_df = load_pickle("simulations_portfolio.obj")

    print(df1, df2, df3, portfolio_df)

    logret = lambda df: np.log((1 + df.capital_ret).cumprod())

    plt.plot(logret(portfolio_df), label="portfolio")
    plt.plot(logret(df1), label="df1")
    plt.plot(logret(df2), label="df2")
    plt.plot(logret(df3), label="df3")
    plt.legend(loc="upper left")
    plt.show()
    plt.close()

    # portfolio_df = load_pickle("simulations_portfolio.obj")
    # df1_, df2_, df3_ = load_pickle("simulations_inst_vol_targeting.obj")
    # df1__, df2__, df3__ = load_pickle("simulations.obj")

    # print("df1, df1_, df1__")
    # print(df1, df1_, df1__)
    #
    # print("df2, df2_, df2__")
    # print(df2, df2_, df2__)
    #
    # print("df3, df3_, df3__")
    # print(df3, df3_, df3__)
    #
    # plt.plot(df1.capital, label="df1")
    # plt.plot(df1_.capital, label="df1_")
    # plt.plot(df1__.capital, label="df1__")
    # plt.legend(loc="upper left")
    # plt.show()
    # plt.close()
    #
    # plt.plot(df2.capital, label="df2")
    # plt.plot(df2_.capital, label="df2_")
    # plt.plot(df2__.capital, label="df2__")
    # plt.legend(loc="upper left")
    # plt.show()
    # plt.close()
    #
    # plt.plot(df3.capital, label="df3")
    # plt.plot(df3_.capital, label="df3_")
    # plt.plot(df3__.capital, label="df3__")
    # plt.legend(loc="upper left")
    # plt.show()
    # plt.close()
    #
    # # lambda function to remove days with 0 capital returns. we do this because our index contains weekends, holidays,
    # # and other days where there was no market, so the capital return would have been 0 since we ffill and bfill
    # # the market data during the data pre-processing stage
    nzr = lambda df: df.capital_ret.loc[df.capital_ret != 0].fillna(0)
    #
    # plot_vol(nzr(df1))
    # plot_vol(nzr(df1_))
    # plot_vol(nzr(df1__))
    # plot_vol(nzr(df2))
    # plot_vol(nzr(df2_))
    # plot_vol(nzr(df2__))
    # plot_vol(nzr(df3))
    # plot_vol(nzr(df3_))
    # plot_vol(nzr(df3__))
    #
    # print volatility of capital returns
    print(nzr(df1).std() * np.sqrt(253), nzr(df2).std() * np.sqrt(253), nzr(df3).std() * np.sqrt(253), nzr(portfolio_df).std() * np.sqrt(253))
    # print(nzr(df1_).std() * np.sqrt(253), nzr(df2_).std() * np.sqrt(253), nzr(df3_).std() * np.sqrt(253))
    # print(nzr(df1__).std() * np.sqrt(253), nzr(df2__).std() * np.sqrt(253), nzr(df3__).std() * np.sqrt(253))
