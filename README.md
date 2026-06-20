# AutoTradingPlatform

[![PyPI - Status](https://img.shields.io/pypi/v/shioaji.svg?style=for-the-badge)](https://pypi.org/project/shioaji)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/shioaji.svg?style=for-the-badge)]()

AutoTradingPlatform is a *trading framework* built on top of the [Shioaji API](https://sinotrade.github.io/) and supports **Shioaji 1.5+**. It can trade **stocks, futures, and options**, including single-leg option orders and option combo orders. The framework streamlines the use of the API, making it easy for users to start trading without being familiar with the API's interface specifications. With just two simple steps, users can start trading:

1. Develop a trading strategy.
2. Set the risk tolerance.

When using AutoTradingPlatform, users can execute trades with multiple strategies on a single account, or even on multiple accounts. The platform also supports multi-frequency intraday monitoring, runtime strategy controls such as updating `max_qty`, and position synchronization between broker-side positions and local monitoring records.


- [AutoTradingPlatform](#autotradingplatform)
  - [What AutoTradingPlatform is capable of?](#what-autotradingplatform-is-capable-of)
    - [1. Position control](#1-position-control)
    - [2. Multi-frequency K-bars and technical indicators](#2-multi-frequency-k-bars-and-technical-indicators)
    - [3. Real-time notification](#3-real-time-notification)
    - [4. Stock selection](#4-stock-selection)
    - [5. Historical backtesting](#5-historical-backtesting)
    - [6. Full automation](#6-full-automation)
    - [7. Telegram Bot interactions](#7-telegram-bot-interactions)
    - [8. Stocks, futures, and options orders](#8-stocks-futures-and-options-orders)
    - [9. Runtime position synchronization](#9-runtime-position-synchronization)
  - [Required packages](#required-packages)
  - [Project Structure](#project-structure)
  - [Preparation](#preparation)
    - [Step 1: Create system settings (config.ini)](#step-1-create-system-settings-configini)
    - [Step 2: Create user settings (user env settings)](#step-2-create-user-settings-user-env-settings)
    - [Step 3: Create strategy scripts](#step-3-create-strategy-scripts)
  - [Execute Commands](#execute-commands)
      - [0. GUI control panel](#0-gui-control-panel)
      - [1. Auto trader](#1-auto-trader)
      - [2. Stock selection](#2-stock-selection)
  - [Releases and Contributing](#releases-and-contributing)
  
## What AutoTradingPlatform is capable of?

#### 1. Position control
The system monitors order amount, position size, strategy-level quantity limits, and batch exits. Multiple strategies can run on the same account while each strategy keeps its own position state and `max_qty` controls.

#### 2. Multi-frequency K-bars and technical indicators
In addition to traditional indicators such as MACD, KD, RSI, you can customize indicators based on technical, chip, and fundamental information. Intraday monitoring stores K-bars by frequency in `TradeData.KBars.Freq`, and strategies can monitor multiple frequencies such as `1T`, `5T`, `15T`, `30T`, `60T`, and `1D`.

#### 3. Real-time notification
Receive real-time order/deal messages through Telegram or LINE Notify.

#### 4. Stock selection
By customizing the selection criterias, the stock selection program can be seamlessly integrated with the trading system to place orders. There is no limit to the number of strategies.

#### 5. Historical backtesting
Using historical data and customizable strategy functions, the backtesting framework supports single-frequency and multi-frequency K-bar strategies. Futures-style strategies can prepare all configured `kbarScales` in advance and receive aligned multi-frequency data in strategy callbacks through `**kwargs`.

#### 6. Full automation
Once everything is set up, you can schedule and achieve fully automated trading.

#### 7. Telegram Bot interactions
Read [TelegramBot.md](./TelegramBot.md) for more instructions. Telegram commands can be used to check and update strategy `max_qty` while the monitor is running.

#### 8. Stocks, futures, and options orders
The order layer supports stock, futures, single-option, and option-combo order workflows for Shioaji 1.5+. Option helpers normalize option contract lookup and order creation so strategy code can focus on trading logic.

#### 9. Runtime position synchronization
AutoTradingPlatform can synchronize broker-side positions into local monitoring records. This helps recover monitoring state after manual orders, restarts, fills from outside the strategy loop, or changes made while the bot is running.

## Required packages
Install Shioaji 1.5+ and other packages at a time. The current requirements pin Shioaji to `1.5.3`.
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
  |-- archives                       (legacy tools kept for reference)
  |-- gui.py                         (local GUI control panel and dashboard)
  |-- tasker.py                      (AutoTradingPlatform task execution file)
  ...
```


## Preparation
#### Step 1: Create system settings (config.ini)
Go to ```./lib``` and create a ```config.ini``` file, which has 6 main sections: 

```ini
[ACCOUNT] # user names
USERS = your_env_file_name
LOG_LEVEL = DEBUG

[COST] # Account trading costs for stocks/futures/options
STOCK_FEE_RATE = 0.001425
FUTURES_FEE_TXF = 100
FUTURES_FEE_MXF = 100
FUTURES_FEE_TMF = 100
FUTURES_FEE_TXO = 100

[DATA]
DATA_PATH = your/path/to/save/datasets

[DB] # optional, can be left blank
DB_ENGINE = mysql+pymysql/postgresql
REDIS_HOST = your_redis_host
REDIS_PORT = your_redis_port
REDIS_PWD = your_redis_password
DB_HOST = your_DB_host
DB_PORT = your_DB_port
DB_USER = your_DB_user_name
DB_PWD = your_DB_password
DB_NAME = your_DB_schema_name

[NOTIFY] # optional, can be left blank, PLATFORM = Telegram/Line
PLATFORM = Telegram
TELEGRAM_TOKEN = {"user_name": "your_Telegram_Notify_token"}
TELEGRAM_CHAT_ID = {"user_name": "your_Telegram_Chat_ID"}
LINE_TOKEN = your_LINE_Notify_token

[STRATEGY]
MonitorFreq = 5      # monitor frequency (seconds per loop)

[CRAWLER] # determine what K-bar frequency data you want for monitor/backtest
SCALES = 1T, 5T, 15T, 30T, 60T, 1D
```

#### Step 2: Create user settings (user env settings)
It is necessary to create user settings for each trading account before starting the auto trader. Start the local GUI:

```bash
streamlit run gui.py
```

Then open the Streamlit URL shown in the terminal, usually http://localhost:8501, and go to the `使用者專區` tab. You can add, edit, or delete account settings there. The settings will be saved to the `user_settings` database table.

The old `python run.py -TASK create_env` Flask setup page has been archived under `archives/legacy_create_env_app`. Use the GUI `使用者專區` for new account setup and maintenance.

#### Step 3: Create strategy scripts
Go to ```./trader/scripts``` and follow the [instructions](./trader/scripts/README.md) to create your own trading strategy before starting the auto-trader. Strategy modules can define:

```python
scale = '1T'
kbarScales = ['1T', '15T']

def add_features(df, scale='1T'):
    ...

def examineOpen(trade, price=None, **kwargs):
    K15min = kwargs.get('K15T')
    ...
```

During backtesting, `scale` is the main loop frequency and `kbarScales` determines which frequencies are prepared and aligned. During intraday monitoring, `TradeData.KBars.Freq[scale]` remains a pandas DataFrame for each frequency.

## Execute Commands
Open a terminal and execute the following task command type:

#### 0. GUI control panel
Start the local GUI dashboard and control panel:

```bash
streamlit run gui.py
```

The GUI supports per-account trader startup, pause/resume/stop commands, status monitoring, recent log display, current position display, runtime `max_qty` updates, and `UserSettings` maintenance.

If you need to bind the server explicitly, for example on Windows local use:

```bash
streamlit run gui.py --server.address 127.0.0.1 --server.port 8501
```

To stop the GUI, go back to the terminal running Streamlit and press `Ctrl+C`. If the GUI was started in the background, stop the Streamlit process from Task Manager, or terminate the corresponding Python/Streamlit PID.

#### 1. Auto trader  
parameter ACCT: account_name defined by users
```
python run.py -TASK auto_trader -ACCT YourAccountName
```

#### 2. Stock selection  
This task will run stock data crawler (using API) and then select stock. Details of establishing selection scripts see the [instructions](./trader/scripts/README.md).
```
python run.py -TASK update_and_select_stock
```

## Releases and Contributing
AutoTradingPlatform has a 7-day release cycle, any updates will be committed by each Friday (git commits are not included).
