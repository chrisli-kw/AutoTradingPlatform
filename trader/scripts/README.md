# Script Design Instructions

The instruction document regarding script design has 4 parts:

- [Script Design Instructions](#script-design-instructions)
  - [Long/Short Strategies](#longshort-strategies)
    - [Step 1: add files](#step-1-add-files)
    - [Step 2: add strategy scripts](#step-2-add-strategy-scripts)
  - [K-Bar Features](#k-bar-features)
    - [Step 1: add file](#step-1-add-file)
    - [Step 2: add feature scripts](#step-2-add-feature-scripts)
  - [Stock Selection](#stock-selection)
    - [Step 1: add file](#step-1-add-file-1)
    - [Step 2: add selection scripts](#step-2-add-selection-scripts)
  - [Backtesting Script Example](#backtesting-script-example)


## Long/Short Strategies

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

## K-Bar Features
After developing your trading strategies, you should also develop a feature script file to generate features (or technical indicators) for each strategy. Otherwise, the strategies would run with errors (lack of features). Similar to [Long/Short Strategies](#longshort-strategies), feature scripts is added by 2 steps:

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


## Stock Selection
Stock selection is also an important part of auto-trading, since traders cannot trade without target securities. You can also define your own scripts only by 2 steps:

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


## Backtesting Script Example
```lua
.  
  |-- trader
    |-- scripts
      |-- __BacktestScripts__.py   <---
```
Create a python file named ```__BacktestScripts__.py``` and then add scripts to it. For detailed script writing instructions, please refer to the [sample backtesting script](../../docs/script%20samples/backtest_sample.py)