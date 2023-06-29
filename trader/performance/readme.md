# AutoTradingPlatform Backtester

This backtesting module is a framework designed for building flexible trading strategies, and it supports backtesting with different candlestick frequencies. Therefore, whether it is stock or futures data, as long as you provide the correct data and script, you can use this framework to perform backtesting and evaluate the performance of your strategy.

- [AutoTradingPlatform Backtester](#autotradingplatform-backtester)
  - [Customize Backtest Scripts](#customize-backtest-scripts)
  - [Data Prepareation](#data-prepareation)
    - [1. Database](#1-database)
    - [2. Local Directories](#2-local-directories)
      - [Default data names:](#default-data-names)
      - [Default path to the datasets:](#default-path-to-the-datasets)
  - [Backtest Results](#backtest-results)
  - [Backtesting Methodology](#backtesting-methodology)

## Customize Backtest Scripts
The backtest scripts that the framework supports are as follows:
1. dataset features
2. stock selection conditions
3. conditions to open a position
4. conditions to close a position (Supports closing a position at a time, or many times.)
5. portfolio limit
6. units of the security to open a position
7. trade by Cash/Margin Leverage

## Data Prepareation
Datasets are stored either in a *database* or in local directories.
### 1. Database
Based on the kbar frequency you select, we have 3 tables for backtesting:
1. KBarData1D
2. KBarData60T
3. KBarData30T

### 2. Local Directories
#### Default data names: 
1. stock data (monthly): ```{YYYY}-{MM}-stock_data_{kbar_freq}```
2. stock data (daily): ```{YYYY}-{MM}-{DD}-stock_data_{kbar_freq}```
3. futures data: ```futures_data_1T```  

PS: The stock datasets are merged at the end of each month (after daily crawler finishes). Therefore, there are 2 kinds of file names of the stock data.

#### Default path to the datasets:
1. stock data (monthly): ```./data/Kbars/{kbar_freq}/{YYYY}-{MM}-stock_data_{kbar_freq}.pkl```
2. stock data (daily): ```./data/Kbars/{kbar_freq}/{YYYY}-{MM}-{DD}-stock_data_{kbar_freq}.pkl```
3. futures data: ```./data/Kbars/futures_data_1T.pkl```

## Backtest Results
After running the backtest module, you can get the following results:
1. Summary
2. Transaction Detail
3. table of winning rates under each trading condition
4. Account Balance Plot
5. Accumulated Profit
6. Accumulated Loss
7. Changes in Opens & Closes
8. Changes in Portfolio Control
The results above can be exported as a .html file.


## Backtesting Methodology
Once you have a basic strategy concept, you can write the strategy framework in the form of a Python class script. This class object should include the [7 kinds of Customize Backtest Scripts](#customize-backtest-scripts) above. After completing the script, create a BackTester object, then call the set_scripts function to configure the script, and you can begin the backtesting process.


