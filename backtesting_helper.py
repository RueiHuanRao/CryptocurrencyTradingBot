# -*- coding: utf-8 -*-

from data_preprocessing import DataPreprocessing
from traditional_strategies import *  # noqa

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os


plt.style.use("seaborn-v0_8-darkgrid")
dp = DataPreprocessing(symbol="BTC-USD")


def plot_buy_sell_signals(df: pd.DataFrame, file_name, save_image, symbol):

    # Scatter plot for signals
    plt.scatter(df.index[df['Signal'] == 1], df['Close'][df['Signal'] == 1],  # noqa
                color='blue', marker='*', s=100, label='Buy Signal')
    plt.scatter(df.index[df['Signal'] == -1], df['Close'][df['Signal'] == -1],  # noqa
                color='red', marker='*', s=100, label='Sell Signal')
    plt.plot(df.index, df['Close'], color='grey', label='Close')  # noqa

    plt.tight_layout()

    if save_image:
        folder = rf".\Project\strategy_logs\{symbol}"
        path = os.path.join(folder, file_name)
        plt.savefig(path)
    else:
        plt.show()


def select_date(
        df: pd.DataFrame,
        start: datetime = None,
        end: datetime = None,
        resample_interval: str = None
):

    assert isinstance(df.index, pd.DatetimeIndex), \
           "Please convert the index to datetime first"

    df = df.copy()

    if start and end:
        timezone = pytz.timezone("UTC")
        start = timezone.localize(start)
        end = timezone.localize(end)
        df = df.loc[start:end].dropna(inplace=True)

    # resample the data
    if resample_interval:
        df["Open"] = df.Open.resample(resample_interval).first()
        df["Close"] = df.Close.resample(resample_interval).last()
        df["High"] = df.High.resample(resample_interval).max()
        df["Low"] = df.Low.resample(resample_interval).min()
        df["Volume"] = df.Volume.resample(resample_interval).sum()
        df["Trades"] = df.Trades.resample(resample_interval).sum()

    df.dropna(inplace=True)

    return df


def data_for_testing(
        symbol: str = "BTCEUR",
):
    dfs = pd.DataFrame()
    for i in range(13):
        date = (datetime(2022, 6, 1) + timedelta(days=31*i)).strftime("%Y-%m")
        file = rf"DataCollection\{symbol}\Binance\{symbol}-1m-{date}.csv"
        df = dp.load_data(file)
        dfs = dfs._append(df)

    return dfs


def data_for_optimisation(
        symbol: str = "BTCEUR",
        resample_interval: str = None,
):

    dfs = pd.DataFrame()
    for i in range(12):
        date = (datetime(2021, 5, 1) + timedelta(days=31*i)).strftime("%Y-%m")
        file = rf"DataCollection\{symbol}\Binance\{symbol}-1m-{date}.csv"
        df = dp.load_data(file)
        dfs = dfs._append(df)

    dfs = dfs.sort_index()

    if resample_interval:
        df_resampled = select_date(
            dfs,
            resample_interval=resample_interval
        )

        return dfs, df_resampled
    return dfs


def log_strategy(
        df: pd.DataFrame,
        symbol: str,
        my_strategy: object,
        stats: object
):
    """
    - symbol
    - dataset
        1. time period
        2. time interval

    - strategy function:
        1. name
        2. doc description

    - optimised args
    - optimised results
    - buy / sell timings
    """

    folder = rf".\Project\strategy_logs\{symbol}"
    file_name = my_strategy.__name__ + ".txt"
    path = os.path.join(folder, file_name)
    os.makedirs(folder, exist_ok=True)

    with open(path, "a") as f:
        # title
        f.write(f"Symbol: {symbol}, Strategy: {my_strategy.__name__}\n\n")

        # strategy description
        f.write(f"strategy desc:\n    {my_strategy.__doc__.strip()}\n\n")  # noqa

        # dataset time frame
        start = df.index[0]
        end = df.index[-1]
        f.write(f"Time period: {start} to {end}\n\n")
        f.write(str(df) + "\n\n")

        # optimisation
        ## parameters  # noqa
        f.write("optimised args:\n")
        f.write("    " + str(stats._strategy) + "\n\n")

        ## results  # noqa
        f.write(str(stats) + "\n\n")

        ## buy / sell timings  # noqa
        f.write("buy / sell timings:\n")
        f.write(str(df[(df["Signal"] == 1) | (df["Signal"] == -1)]) + "\n\n")  # noqa
        f.write("# " + '=' * 100 + " #")
        f.write("\n\n")
