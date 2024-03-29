# -*- coding: utf-8 -*-

from backtesting_helper import data_for_optimisation, data_for_testing  # noqa
from backtesting_helper import plot_buy_sell_signals, log_strategy  # noqa
from traditional_strategies import *  # noqa

from backtesting import Backtest, Strategy

import time


symbol = "BTCEUR"
resample_interval = "30min"
df, df_resampled = data_for_optimisation(symbol, resample_interval)
my_strategy = strategy_3  # noqa


class StrategyClass(Strategy):

    bear_bull_window: int = 5
    slope_threshold: float = 0.05
    short_rsi_window: int = 15
    long_rsi_window: int = 14
    short_upper: int = 64
    short_lower: int = 32
    long_upper: int = 65
    long_lower: int = 33

    def init(self):

        data = self.I(
            my_strategy,  # noqa
            self.data.df,
            self.bear_bull_window,
            self.slope_threshold,
            self.short_rsi_window,
            self.long_rsi_window,
            self.short_upper,
            self.short_lower,
            self.long_upper,
            self.long_lower
        )
        self.signal = data[-1]

    def next(self):

        # sell
        if self.signal[-1] == -1:
            self.position.close()

        # buy
        elif self.signal[-1] == 1:
            self.buy()


bt = Backtest(
    df_resampled,
    StrategyClass,
    cash=100_000,
    commission=.002,
)


def optim_func(series):

    if series["# Trades"] < 10:
        return -1  # indicate a loss

    # res = series["Equity Final [$]"] / series["Exposure Time [%]"]
    res = series["Equity Final [$]"]

    return res


time_start = time.perf_counter()

stats, heatmap = bt.optimize(
    bear_bull_window=range(14, 64, 7),
    slope_threshold=[0.15, 0.17, 0.2],
    short_rsi_window=14,
    long_rsi_window=14,
    short_upper=range(35, 65, 2),
    short_lower=range(15, 35, 2),
    long_upper=range(54, 75, 2),
    long_lower=range(25, 55, 2),
    # maximize="Equity Final [$]",
    maximize=optim_func,
    # constraint=lambda x: x.upper > x.lower,
    return_heatmap=True,
    # max_tries=0.5
)

print(stats)
print()
print("#" * 44)
print(f"# Time comsume: {time.perf_counter() - time_start} seconds #\n")
print("#" * 44)
print()

print(stats._strategy)

# Optimisation result
df_signal_opt = my_strategy(  # noqa
    df_resampled,
    bear_bull_window=stats._strategy.bear_bull_window,
    slope_threshold=stats._strategy.slope_threshold,
    short_rsi_window=stats._strategy.short_rsi_window,
    long_rsi_window=stats._strategy.long_rsi_window,
    short_upper=stats._strategy.short_upper,
    short_lower=stats._strategy.short_lower,
    long_upper=stats._strategy.long_upper,
    long_lower=stats._strategy.long_lower
)

log_strategy(df_signal_opt, symbol, my_strategy, stats)

i = 1
file_name = my_strategy.__name__ + f"_{i}.png"
plot_buy_sell_signals(df_signal_opt, file_name, save_image=False, symbol=symbol)  # noqa
i += 1

# ========================================================================== #

# test result
# df_test = data_for_testing(symbol)
# df_signal_test = my_strategy(  # noqa
#     df_test,
#     ewm_span=stats._strategy.ewm_span,
#     rsi_window=stats._strategy.rsi_window,
#     upper=stats._strategy.upper,
#     lower=stats._strategy.lower)

# plot_buy_sell_signals(df_signal_test)
