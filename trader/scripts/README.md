# Script Design Instructions

Overall, this framework is divided into two main parts: the trading system and the backtesting system. Whether it's the trading or backtesting system, both consist of three main components:
1. Indicator Features
2. Target Selection
3. Open/Close Positions

Therefore, when writing scripts, they can be categorized into these three types. Whether you are running the trading system or the backtesting system, these three types of scripts will be invoked to form a complete trading strategy.

- [Script Design Instructions](#script-design-instructions)
  - [Indicator Features](#indicator-features)
    - [Step 1: add file](#step-1-add-file)
    - [Step 2: add feature scripts](#step-2-add-feature-scripts)
  - [Target Selection](#target-selection)
    - [Step 1: add file](#step-1-add-file-1)
    - [Step 2: add selection scripts](#step-2-add-selection-scripts)
  - [Open/Close Positions](#openclose-positions)
    - [Step 1: add files](#step-1-add-files)
    - [Step 2: add strategy scripts](#step-2-add-strategy-scripts)
  - [Backtesting Script Example](#backtesting-script-example)


## Indicator Features
After developing your trading strategies, you should also develop a feature script file to generate features (or technical indicators) for each strategy. Otherwise, the system would run with errors (lack of features). Sometimes, some of the features are only used for backtesting and are not used for trading. You can further divide the feature script into 2 kinds of versions: one is for trading and the other is for backtesting.

### Step 1: add file
Add ```features.py``` to ```./trader/scripts/```.

```lua
.
  |-- trader
    |-- scripts
      |-- features.py <--
```

### Step 2: add feature scripts
The script should be an class object, see the [sample code](../../docs/script%20samples/features.py) for more details.


## Target Selection
Target selection is also an important part of auto-trading, since traders cannot trade without target securities. You can also define your own scripts only by 2 steps:

### Step 1: add file
Add ```conditions.py``` to ```./trader/scripts/```.

```lua
.
  |-- trader
    |-- scripts
      |-- conditions.py <--
```

### Step 2: add selection scripts
The script should be an class object, see the [sample code](../../docs/script%20samples/conditions.py) for more details.


## Open/Close Positions
In thsi part, the open/close script can not only have conditions to open/close a position, but also have other conditions for portfolio limit, trading quantity, ..., etc. Similar to [Indicator Features](#indicator-features), feature scripts is added by 2 steps:

### Step 1: add files
Add ```StrategySet.py``` to ```./trader/scripts/``` as a strategy module.

```lua
.
  |-- trader
    |-- scripts
      |-- StrategySet.py  <--
```

### Step 2: add strategy scripts
The script should be an class object, see the [sample code](../../docs/script%20samples/StrategySet.py) for more details.

<u>Be sure to update self.STRATEGIES, self.Funcs, and self.QuantityFunc (if exists) before running AutoTradingPlatform</u>



## Backtesting Script Example
```lua
.  
  |-- trader
    |-- scripts
      |-- backtest_config.py   <---
```
Create a python file named ```backtest_config.py``` and then integrade the above 3 kinds of scripts to it. For detailed script writing instructions, please refer to the [sample backtesting script](../../docs/script%20samples/backtest_sample.py)