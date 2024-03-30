# Cryptocurrency Trading Bot
A project that using traditional (rule-based) strategies and RL algorithms to trade cryptocurrencies.

A brief overview of each part has been shown below.

![project diagram](<Photos\Trading Bot diagram.png>)

---
## Data Preprocessing
This part involves fetching historical market data from Binance or another source and preparing it for analysis. This includes tasks such as cleaning the data, handling missing values, and formatting it into a suitable structure for analysis.

## Feature Engineering
In feature engineering, new features will be constructed, including some common technical indicators, customised features, and any other data points that are predictive of future price movements.

## Traditional Strategies
Rule-based strategies will be implemented in this section, such as simple moving average crossover, or relative strength index based strategies (RSI), such as buy when RSI < 30 and sell when RSI > 70... ect.

## Reinforcement Strategy
Reinforcement learning-based strategies involve training a model to make trading decisions based on rewards received from its actions. This approach allows the bot to adapt to changing market conditions and potentially discover more complex patterns that traditional strategies might miss.

## Strategy Backtesting
Before deploying the trading bot in a live environment, it's essential to backtest the strategies using historical data to evaluate their performance. This involves simulating trades using past market data to see how well the strategies would have performed in real-world conditions.
