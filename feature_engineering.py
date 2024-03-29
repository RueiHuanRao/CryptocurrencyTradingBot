# -*- encoding: "utf-8" -*-

# import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import pandas as pd
import ta
from finta import TA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
import joblib
import numpy as np
from collections import deque

# warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-dark-palette")


@dataclass
class FeatureEngineering:
    """
    A class for performing feature engineering on cryptocurrency market data.

    Attributes:
        timestamp (str): The current timestamp in YYYYMMDD format.
        load_to_fit (bool): Indicates whether data should be loaded for fitting.

    Methods:
        _print: Print a message with decorative borders.
        _dec: A decorator for timing method execution.
        linear_fit: Calculate the linear regression slope.
        lr_bearish_bullish_signal: Identify bullish/bearish signals based on linear regression slope.
        add_rsi: Add the Relative Strength Index (RSI) to a DataFrame.
        ... etc.
    """  # noqa

    timestamp: datetime = datetime.now().strftime("%Y%m%d")
    load_to_fit: bool = False

    def _print(self, str) -> None:

        print()
        print("#", "-" * (len(str)), "#")
        print(f"# {str} #")
        print("#", "-" * (len(str)), "#", "\n")

    def _dec(func):

        def wrap(self, *args, **kwargs):
            begin = time.perf_counter()
            # self._print(f"Processing: {func.__name__}")
            res = func(self, *args, **kwargs)
            self._print(
                f"{func.__name__} finished in "
                + f"{round(time.perf_counter() - begin, 4)}s"
            )

            return res

        return wrap

    def linear_fit(
            self,
            x: np.ndarray,
            y: np.ndarray
    ):

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        denominator = np.sum((x-x_mean)**2)
        numerator = np.sum((x-x_mean)*(y-y_mean))

        slope = numerator/denominator
        # intercept = y_mean - slope*x_mean

        return slope

    def lr_bearish_bullish_signal(
            self,
            df: pd.DataFrame,
            lookback: int = 5,
            slope_threshold: float = 0.001
    ):
        """
        using the past few prices to calculate the slope
        if slope > 0 -> bullish
        if slope < 0 -> bearish
        """

        df = df.copy()

        length = df.shape[0]
        slope_vals = [0] * length

        x = np.asarray([i for i in range(1, lookback+1)])
        q = deque(maxlen=int(lookback))

        for i in range(length):
            q.append(df.iloc[i].Close)

            if i >= lookback-1:
                y = np.asarray(q)

                # cal slope
                slope = self.linear_fit(x, y)

                slope_vals[i] = slope

        def bearish_bullish(row):
            # Bullish
            if row["slope"] > slope_threshold:
                return 1
            # Bearish
            elif row["slope"] < -slope_threshold:
                return -1
            # no trend
            else:
                return 0

        df["slope"] = slope_vals
        df["Bearish_Bullish_Signal"] = df.apply(bearish_bullish, axis=1)

        return df

    def add_rsi(self, df: pd.DataFrame, window=20) -> pd.DataFrame:

        df = df.copy()

        assert "Close" in df.columns

        if df.shape[0] >= window:
            df[f"RSI_{window}"] = ta.momentum.rsi(df["Close"], window=window)

        return df

    def add_pct(
            self,
            df: pd.DataFrame,
            lags: list = range(2, 11)
    ) -> pd.DataFrame:

        df = df.copy()
        assert "Close" in df.columns
        assert len(lags) > 0

        df["PCT1"] = df.index.map(df.Close.pct_change())

        lagging_days = lags
        for i in lagging_days:
            if df.shape[0] >= i:
                df[f"PCT{i}"] = df.index.map(df["PCT1"].shift(i - 1))

        return df

    def add_MAs(
            self,
            df: pd.DataFrame,
            MAs: list = [5, 20, 30, 60]
    ) -> pd.DataFrame:

        df = df.copy()
        assert "Close" in df.columns

        mas = MAs
        for ma in mas:
            if df.shape[0] >= ma:
                df[f"MA{ma}"] = df["Close"].rolling(window=ma).mean()

        return df

    # Bollinger Bands
    def add_bollinger_bands(
            self,
            df: pd.DataFrame,
            window: int = 20,
            window_dev: int = 2
    ) -> pd.DataFrame:

        df = df.copy()
        assert "Close" in df.columns

        if df.shape[0] >= window:
            df["Bollinger_high"] = ta.volatility.bollinger_hband(
                df["Close"], window=window, window_dev=window_dev
            )
            df["Bollinger_low"] = ta.volatility.bollinger_lband(
                df["Close"], window=window, window_dev=window_dev
            )

        return df

    # MACD
    def add_macd(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:

        df = df.copy()
        assert "Close" in df.columns

        # Get the 26-day EMA of the closing price
        k = df["Close"].ewm(span=12, adjust=False, min_periods=12).mean()

        # Get the 12-day EMA of the closing price
        d = df["Close"].ewm(span=26, adjust=False, min_periods=26).mean()

        # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
        macd = k - d

        # Get the 9-Day EMA of the MACD for the Trigger line
        macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()

        # Calculate the difference between the MACD
        # Trigger for the Convergence/Divergence value
        macd_h = macd - macd_s

        # Add all of our new values for the MACD to the dataframe
        df["MACD"] = df.index.map(macd)
        df["MACD_H"] = df.index.map(macd_h)
        df["MACD_S"] = df.index.map(macd_s)

        return df

    def add_obv(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:

        df = df.copy()
        df["OBV"] = TA.OBV(df)

        return df

    def standardisation(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        scaler = StandardScaler()
        index = df.index
        cols = df.columns

        if self.load_to_fit:
            scaler_fit = joblib.load(
                rf"{self.data_preprocessing_dir}/"
                + rf"standardscaler-{self.timestamp}.pkl"
            )
        else:
            scaler_fit = scaler.fit(df)
            joblib.dump(
                scaler_fit,
                rf"{self.data_preprocessing_dir}/"
                + rf"standardscaler-{self.timestamp}.pkl"
            )

        df_scaled = scaler_fit.transform(df)
        df = pd.DataFrame(df_scaled, index=index, columns=cols)

        return df

    def normalisation(self, df: pd.DataFrame) -> pd.DataFrame:

        # FIXME: need to rotate the data for normalisation first

        df = df.copy()
        if "RSI" in df.columns:
            df_rsi = df["RSI"]
            del df["RSI"]

        scaler = Normalizer()
        index = df.index
        cols = df.columns

        if self.load_to_fit:
            scaler_fit = joblib.load(
                rf"{self.data_preprocessing_dir}/"
                + rf"normalisescaler-{self.timestamp}.pkl"
            )
        else:
            scaler_fit = scaler.fit(df)
            joblib.dump(
                scaler_fit,
                rf"{self.data_preprocessing_dir}/"
                + rf"normalisescaler-{self.timestamp}.pkl"
            )

        df_scaled = scaler_fit.transform(df)
        df = pd.DataFrame(df_scaled, index=index, columns=cols)

        if "RSI" in df.columns:
            df["RSI"] = df.index.map(df_rsi)

        return df

    def min_max_scale(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        # --------------------------------------------------------
        # avoid transforming the data
        if "RSI" in df.columns:
            df_rsi = df["RSI"]
            del df["RSI"]
        # --------------------------------------------------------

        scaler = MinMaxScaler((0, 1))
        index = df.index
        cols = df.columns

        if self.load_to_fit:
            scaler_fit = joblib.load(
                rf"{self.data_preprocessing_dir}/"
                + rf"minmaxscaler-{self.timestamp}.pkl"
            )
        else:
            scaler_fit = scaler.fit(df)
            joblib.dump(
                scaler_fit,
                rf"{self.data_preprocessing_dir}/"
                + rf"minmaxscaler-{self.timestamp}.pkl"
            )

        df_scaled = scaler_fit.transform(df)
        df = pd.DataFrame(df_scaled, index=index, columns=cols)

        # --------------------------------------------------------
        if "RSI" in df.columns:
            df["RSI"] = df.index.map(df_rsi)
        # --------------------------------------------------------

        return df

    def pca(self, df, n_components: int) -> pd.DataFrame:

        df = df.copy()
        pca = PCA(n_components=n_components)

        # keep normalised 'Close' column for the env to calculate rewards
        index = df.index
        df_Close = df["Close"]
        del df["Close"]

        if self.load_to_fit:
            pca_fit = joblib.load(
                rf"{self.data_preprocessing_dir}/pca.pkl"
            )
        else:
            pca_fit = pca.fit(df)
            self._print(
                rf"PCA Total Explained Variance ({n_components}): "
                + rf"{round(sum(pca.explained_variance_ratio_), 4)}"
            )
            joblib.dump(pca_fit, rf"{self.data_preprocessing_dir}/pca.pkl")

        data_transformed = pca_fit.transform(df)
        df_transformed = pd.DataFrame(data_transformed, index=index)

        df_transformed["Close"] = df_transformed.index.map(df_Close)

        return df_transformed

    def rsi_divergence(
            self,
            df: pd.DataFrame,
            look_back: int = 10,
            rsi_threshold: tuple = (30, 70)
    ) -> pd.DataFrame:
        """
        ref:
            How to Calculate RSI Divergence in Excel [Easy to Backtest]
            : https://www.youtube.com/watch?v=1dOloHA765s
        """

        df = df.copy()
        assert "Close" in df.columns
        assert "RSI" in df.columns

        data_len = df.shape[0]

        RSI_LOW = rsi_threshold[0]
        RSI_HIGH = rsi_threshold[1]

        df["RSI"] = ta.momentum.rsi(df["Close"], window=20)

        close = df["Close"]
        rsi = df.loc[:, "RSI"]
        price_new_low = [False] * data_len
        rsi_low = [False] * data_len
        price_new_high = [False] * data_len
        rsi_high = [False] * data_len

        for i in range(data_len):

            if i >= look_back:
                pre = i - 1
                # Bullish RSI
                price_new_low[i] = \
                    (close.iloc[pre] < min(close.iloc[(i-look_back):pre])) \
                    & (close.iloc[pre] < close.iloc[i])
                rsi_low[i] = \
                    (rsi.iloc[pre] > min(rsi.iloc[(i-look_back):pre])) \
                    & (rsi.iloc[pre] < rsi.iloc[i]) \
                    & (rsi.iloc[pre] < RSI_LOW)

                # Bearish RSI
                price_new_high[i] = \
                    (close.iloc[pre] > max(close.iloc[(i-look_back):pre])) \
                    & (close.iloc[pre] > close.iloc[i])
                rsi_high[i] = \
                    (rsi.iloc[pre] < max(rsi.iloc[(i-look_back):pre])) \
                    & (rsi.iloc[pre] > rsi.iloc[i]) \
                    & (rsi.iloc[pre] > RSI_HIGH)

        df["PRICE_NEW_LOW"] = price_new_low
        df["RSI_LOW"] = rsi_low
        df["PRICE_NEW_HIGH"] = price_new_high
        df["RSI_HIGH"] = rsi_high

        df["Bullish_Divergence"] = df.apply(
            lambda row: int((row["PRICE_NEW_LOW"] == True) & (row["RSI_LOW"] == True)), # noqa
            axis=1
        )
        df["Bearish_Divergence"] = df.apply(
            lambda row: int((row["PRICE_NEW_HIGH"] == True) & (row["RSI_HIGH"] == True)), # noqa
            axis=1
        )

        del df["PRICE_NEW_LOW"], df["RSI_LOW"]
        del df["PRICE_NEW_HIGH"], df["RSI_HIGH"]
        del df["RSI"]

        return df

    def engulfing_signals(
            self,
            df: pd.DataFrame,
            body_diff_min: float = 0.003,
            engulfing_coeff_upper: float = +5e-5,
            engulfing_coeff_lower: float = +5e-5
    ) -> pd.DataFrame:

        df = df.copy()
        assert "Close" in df.columns
        assert "Open" in df.columns

        df["Body_diff"] = df.apply(
            lambda row: abs(row["Close"] - row["Open"]), axis=1
        )
        df["Open_pre"] = df["Open"].shift(1)
        df["Close_pre"] = df["Close"].shift(1)
        df["Body_diff_pre"] = df["Body_diff"].shift(1)

        def engulfing_conditions(df: pd.DataFrame) -> int:

            # Bearish Engulfing Pattern
            if (df["Body_diff"] > body_diff_min) \
                & (df["Body_diff_pre"] > body_diff_min) \
                & (df["Open_pre"] < df["Close_pre"]) \
                & (df["Open"] > df["Close"]) \
                & (df["Open"] - df["Close_pre"] >= engulfing_coeff_upper) \
                & (df["Open_pre"] - df["Close"] >= engulfing_coeff_lower): # noqa

                return 1

            # Bullish Engulfing Pattern
            elif (df["Body_diff"] > body_diff_min) \
                & (df["Body_diff_pre"] > body_diff_min) \
                & (df["Open_pre"] > df["Close_pre"]) \
                & (df["Open"] < df["Close"]) \
                & (df["Close_pre"] - df["Open"] >= engulfing_coeff_lower) \
                & (df["Close"] - df["Open_pre"] >= engulfing_coeff_upper): # noqa

                return 2

            else:
                return 0

        df["Engulfing_signal"] = df.apply(engulfing_conditions, axis=1)

        del df["Open_pre"], df["Close_pre"]
        del df["Body_diff_pre"], df["Body_diff"]

        return df

    def atr(
        self,
        df: pd.DataFrame,
        window: int = 14
    ) -> pd.DataFrame:
        ...
        """
        for TPSL
        """

        df = df.copy()
        assert "Close" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns

        if df.shape[0] >= window:
            df["ATR"] = ta.volatility.average_true_range(
                df["High"],
                df["Low"],
                df["Close"],
                window=window,
                fillna=False
            )

        return df

    def hammer_signals(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        assert "Close" in df.columns
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns

        df["Range"] = df["High"] - df["Low"]

        def hammer_conditions(df: pd.DataFrame) -> int:
            # to identify the hammer pattern
            fibonacci_ratio = 0.382

            # bullish hammer pattern
            if (df["Close"] > df["Open"]) \
                & (df["Open"] > (df["High"] - fibonacci_ratio * df["Range"])): # noqa
                return 2
            # bearish hammer pattern
            elif (df["Close"] < df["Open"]) \
                & (df["Open"] < (df["Low"] + fibonacci_ratio * df["Range"])): # noqa
                return 1
            else:
                return 0

        df["Hammer_signal"] = df.apply(hammer_conditions, axis=1)

        del df["Range"]

        return df

    def vwma(
            self,
            df: pd.DataFrame,
            window: int = 3
    ):

        df = df.copy()
        assert "Close" in df.columns
        assert "Volume" in df.columns

        if df.shape[0] >= window:
            close = np.array(df["Close"])
            volume = np.array(df["Volume"])
            vwma = [0] * df.shape[0]

            for i in range(window-1, df.shape[0]):
                vwma[i] = sum(close[i-window+1:i+1] * volume[i-window+1:i+1]) \
                    / sum(volume[i-window+1:i+1])

            df["VWMA"] = vwma

        return df

    # TODO: should be modified afterwards
    def vwap(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        assert "Close" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns
        assert "Volume" in df.columns

        close = np.array(df["Close"])
        high = np.array(df["High"])
        low = np.array(df["Low"])
        average_price = (high + low + close)/3
        volume = np.array(df["Volume"])
        agg_avg_price_vol = average_price * volume
        vwap = agg_avg_price_vol / volume

        df["VWAP"] = vwap

        return df

    def feature_engineering_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df = self.add_MAs(
            df,
            MAs=[5, 20, 30, 60]
        )
        df = self.rsi_divergence(
            df,
            look_back=30,
            rsi_threshold=(30, 70)
        )
        df = self.engulfing_signals(
            df,
            body_diff_min=0.003,
            engulfing_coeff_upper=+5e-5,
            engulfing_coeff_lower=+5e-5
        )
        df = self.hammer_signals(df)
        df = self.atr(df)
        df = self.add_macd(df)
        df = self.add_rsi(df, window=30)
        df = self.add_pct(df, lags=range(2, 11))
        df = self.add_bollinger_bands(df, window=30, window_dev=2)
        df = self.add_obv(df)
        df = self.vwma(df, window=30)

        df.dropna(inplace=True)

        # normalisation
        # df = self.normalisation(df)

        # standardisation
        # df = self.standardisation(df)

        # normalise the features
        df = self.min_max_scale(df)

        # perform PCA
        df = self.pca(df, n_components=1)

        return df


if __name__ == "__main__":
    index = [datetime.now() + timedelta(days=i) for i in range(5)]
    df = pd.DataFrame(
        {
            "datetime": index,
            "Close": [1, 3, 6, 4, 2],
            "Open": [10, 20, 30, 40, 5]}
    )
    df.set_index("datetime", inplace=True)

    fe = FeatureEngineering()
    df = fe.lr_bearish_bullish_signal(df, 3)
    df
