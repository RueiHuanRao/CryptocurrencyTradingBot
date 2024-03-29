# -*- encoding: utf-8 -*-

import pandas as pd
import yfinance as yf
from datetime import datetime
from feature_engineering import FeatureEngineering

fe = FeatureEngineering()


def strategy_DEMO(
        df: pd.DataFrame,
        ewm_span: int = 200,
        rsi_window: int = 20,
        upper: int = 70,
        lower: int = 30
) -> pd.DataFrame:
    """
    #------#
    # DEMO #
    #------#
    """

    df["EMW"] = df.Close.ewm(span=ewm_span, adjust=False).mean()
    df = fe.add_rsi(df, rsi_window)

    # strategy here
    def strategy(row) -> int:

        # oversold -> buy
        if row.RSI < lower and row.Close > row.EMW:
            return 1

        # overbuy -> sell
        elif row.RSI > upper and row.Close < row.EMW:
            return -1

        else:
            return 0

    df["Signal"] = df.apply(strategy, axis=1)

    return df


if __name__ == "__main__":

    symbol = "BTC-USD"
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 1)
    df = yf.download(symbol, start, end)
    df.head(5)
    print(df.shape)
    df
    df = strategy_DEMO(df)
    df[df["Signal"] == 1]
