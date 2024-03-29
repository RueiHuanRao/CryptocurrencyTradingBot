# -*- encoding: "utf-8" -*-

# import warnings
import os
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from feature_engineering import FeatureEngineering
# from feature_engineering import FeatureEngineering

# warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-dark-palette")


@dataclass
class DataPreprocessing(FeatureEngineering):
    """
    A class for downloading and preprocessing cryptocurrency market data.

    Attributes:
        symbol (str): The symbol of the cryptocurrency.
        data_preprocessing_dir (str): The directory path for saving preprocessed data.

    Methods:
        load_data: Load historical market data from a CSV file and preprocess it.
        download_historical_data: Download historical data using Yahoo Finance.
        download_historical_data_ticker: Download historical data for a ticker.
        train_test_split: Split data into training and test sets.
        run: Execute the preprocessing pipeline.
    """  # noqa

    # Class variables
    symbol: str = None

    def __post_init__(self) -> None:

        self._set_dirs()

    def _set_dirs(self) -> None:

        self.data_preprocessing_dir = \
            r"./Project/data_processing_objects/" \
            + rf"{self.symbol}-{self.timestamp}"

        os.makedirs(rf"{self.data_preprocessing_dir}/", exist_ok=True)

    def load_data(
            self,
            path
    ) -> pd.DataFrame:
        """
        Load data downloaded from Binance Web
        """
        cols = [
            "datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close_time",
            "Quote_volume",
            "Trades",
            "Taker_buy_volume",
            "Taker_buy_quote_volume",
            "Ignore"
        ]
        df = pd.read_csv(path)  # noqa
        df.columns = cols
        df["datetime"] = df["datetime"]. \
            apply(lambda row: datetime.fromtimestamp(row / 1000))
        df["Close_time"] = df["Close_time"] \
            .apply(lambda row: datetime.fromtimestamp(row / 1000))

        df.set_index("datetime", inplace=True)

        return df

    # download the stock data
    @FeatureEngineering._dec
    def download_historical_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:

        self.df = yf.download(self.symbol, start=start_date, end=end_date)
        del self.df["Adj Close"]

        # data.columns = data.columns.str.lower()
        self.df.index.rename("datetime", inplace=True)

        # reset datetime format
        self.df.index = pd.to_datetime(self.df.index)

        df = self.df.copy()

        return df

    # for ticker
    @FeatureEngineering._dec
    def download_historical_data_ticker(
        self, period: str = "60d", interval: str = "5m"
    ) -> pd.DataFrame:

        self.df = yf.download(
            tickers=self.symbol, period=period, interval=interval)[
            ["Open", "Close", "High", "Low", "Volume"]
        ]
        # data.columns = data.columns.str.lower()
        self.df.index.rename("datetime", inplace=True)

        # reset datetime format
        self.df.index = pd.to_datetime(self.df.index)

        df = self.df.copy()

        return df

    def train_test_split(self, df: pd.DataFrame, test_ratio: float):

        # split test data
        mid_index = int(df.shape[0] * (1 - test_ratio))
        self.test = df.iloc[mid_index:, :]
        self.train = df.iloc[:mid_index, :]

        # log the datetime
        print(f"Time period (train): \
              {self.train.index[0]} - {self.train.index[-1]}")
        print(f"Time period (test) : \
              {self.test.index[0]} - {self.test.index[-1]}")

    @FeatureEngineering._dec
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self.feature_engineering_pipeline(df)

        print(f"{df.shape = }")

        return df


if __name__ == "__main__":

    # symbol = "AAPL"
    symbol = "BTC-EUR"

    cla = DataPreprocessing(symbol=symbol)
    df = cla.load_data()
    print(df.head())
