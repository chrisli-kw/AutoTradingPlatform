# AutoTradingPlatform

[![PyPI - Status](https://img.shields.io/pypi/v/shioaji.svg?style=for-the-badge)](https://pypi.org/project/shioaji)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/shioaji.svg?style=for-the-badge)]()

AutoTradingPlatform is a *trading framework* built on top of the [Shioaji API](https://sinotrade.github.io/) that supports trading in **stocks, futures, and options**. This framework streamlines the use of the API, making it easy for users to start trading without being familiar with the API's interface specifications. With just two simple steps, users can start trading:

1. Develop a trading strategy.
2. Set the risk tolerance.

When using AutoTradingPlatform, users can execute trades with multiple strategies on a single account, or even on multiple accounts. Additionally, the platform ensures safe order placement by monitoring risk tolerance levels. For example, if the order limit of an account is NT$500,000, buying stops when the cumulative amount of stocks approaching NT$500,000.


- [AutoTradingPlatform](#autotradingplatform)
  - [What AutoTradingPlatform is capable of?](#what-autotradingplatform-is-capable-of)
      - [1. Position control](#1-position-control)
      - [2. Technical indicators](#2-technical-indicators)
      - [3. Real-time notification](#3-real-time-notification)
      - [4. Stock selection](#4-stock-selection)
      - [5. Historical backtesting (to be updated)](#5-historical-backtesting-to-be-updated)
      - [6. Full automation](#6-full-automation)
  - [Required packages](#required-packages)
  - [Project Structure](#project-structure)
  - [Prepareation](#prepareation)
      - [Step 1: Create system settings (config.ini)](#step-1-create-system-settings-configini)
      - [Step 2: Create user settings (.env file)](#step-2-create-user-settings-env-file)
      - [Step 3: Create long/short stragety](#step-3-create-longshort-stragety)
      - [Step 4: Create your own K-bar features](#step-4-create-your-own-k-bar-features)
  - [Execute Commands](#execute-commands)
      - [1. Auto trader](#1-auto-trader)
      - [2. Stock selection](#2-stock-selection)
  - [Releases and Contributing](#releases-and-contributing)
  
## What AutoTradingPlatform is capable of?

#### 1. Position control  
Like humans, the system pays attention to order amount, position management, and batch exit. At the same time, we can execute multiple strategies simultaneously to achieve better position control.

#### 2. Technical indicators  
In addition to traditional indicators such as MACD, KD, RSI, you can also customize the indicators you want based on technical, chip, and fundamental information.

#### 3. Real-time notification
Using LINE Notify notifications to receive real-time order/deal messages.

#### 4. Stock selection
By customizing the selection criterias, the stock selection program can be seamlessly integrated with the trading system to place orders. There is no limit to the number of strategies.

#### 5. Historical backtesting (to be updated)
Using historical data and the customizable strategy function of the trading framework to improve and maximize your potential profits.

#### 6. Full automation
Once everything is set up, you can schedule and achieve fully automated trading.

## Required packages
Install shioaji and other packages at a time
```ini
pip install -r requirements.txt
```

## Project Structure
```lua
.  
  |-- templates
  |-- trader                         (Python trading modules, details as below:)
    |-- indicators                   (trading indicators)
    |-- scripts                      (folder for putting CUSTOMIZED TRADING SCRIPTS)
      |-- ...
    |-- ...
  |-- lib                            (keys, settings, ..., etc)
    |-- ckey                         (has public/private keys to encrypt/decrypt password text)  
    |-- ekey                         (Sinopac ca files for placing orders)  
      |-- 551  
        |--...  
    |-- envs                         (AutoTradingPlatform user settings)  
    |-- schedules                    (schedule task files, ex: *.bat)
    |-- config.ini                   (AutoTradingPlatform system settings)
    |-- 政府行政機關辦公日曆表.csv
    ...
  |--data
    |-- Kbars                        (stores any kinds of stock/futures/options/indexes data)  
    |-- selections                   (selected stocks)  
    |-- stock_pool                   (stores watchlist, order list files for AutoTradingPlatform)  
    |-- ticks                        (futures tick data from TAIFEX)
  |-- create_env.py                  (.env file creator)
  |-- tasker.py                      (AutoTradingPlatform task execution file)
  ...
```


## Prepareation
#### Step 1: Create system settings (config.ini)
Go to ```./lib``` and create a ```config.ini``` file, which has 6 main sections: 

```ini
[ACCOUNT] # user names
USERS = your_env_file_name

[DATA]
DATA_PATH = your/path/to/save/datasets

[DB] # optional, can be left blank
REDIS_HOST = your_redis_host
REDIS_PORT = your_redis_port
REDIS_PWD = your_redis_password
DB_HOST = your_DB_host
DB_PORT = your_DB_port
DB_USER = your_DB_user_name
DB_PWD = your_DB_password
DB_NAME = your_DB_schema_name

[LINENOTIFY] # optional, can be left blank
TOKEN_MONITOR = your_LINE_Notify_token_for_system_monitor
TOKEN_INFO = your_LINE_Notify_token_for_trading_monitor

[STRATEGY]
Long = strategy1     # non-daytrade long strategies
Short = strateg2     # non-daytrade short strategies
LongDT = strategy3   # daytrade long strategies
ShortDT = strategy4  # daytrade short strategies
MonitorFreq = 5      # monitor frequency (seconds per loop)

[SELECT] # includes stock selection strategy names
METHODS = StockSelectStrategy1

[CRAWLER] # determine what K-bar frequency data you want for backtest
SCALES = 1D, 30T, 60T
```

#### Step 2: Create user settings (.env file)
It is necessary to create an env file for a whole-new AutoTradingPlatform. Firstly, run the following command in a terminal:  
```
python run.py -TASK create_env
```  

Then, go to http://127.0.0.1:5000/. Press the "Submit" button after filling out the forms, an env file will be created in ```./lib/envs```

#### Step 3: Create long/short stragety
Go to ```./trader/scripts``` and follow the [instructions](./trader/scripts/readme.md#longshort-strategies) to create your own trading strategy before starting the auto-trader.

#### Step 4: Create your own K-bar features
Go to ```./trader/scripts``` and follow the [instructions](./trader/scripts/readme.md#k-bar-features) to create k-bar feature scripts before starting the auto-trader.


## Execute Commands
Open a terminal and execute the following task command type:

#### 1. Auto trader  
parameter ACCT: account_name defined by users
```
python run.py -TASK auto_trader -ACCT YourAccountName
```

#### 2. Stock selection  
This task will run stock data crawler (using API) and then select stock. Details of establishing selection scripts see the [instructions](./trader/scripts/readme.md#stock-selection).
```
python run.py -TASK update_and_select_stock
```

## Releases and Contributing
AutoTradingPlatform has a 7-day release cycle, any updates will be committed by each Friday (git commits are not included).
