# Script Design Instructions

The instruction document regarding script design has 3 parts:

- [Script Design Instructions](#script-design-instructions)
  - [Long/Short Strategies](#longshort-strategies)
    - [Step 1: add files](#step-1-add-files)
    - [Step 2: add long strategy scripts](#step-2-add-long-strategy-scripts)
    - [Step 3: add short strategy scripts](#step-3-add-short-strategy-scripts)
  - [K-Bar Features](#k-bar-features)
    - [Step 1: add file](#step-1-add-file)
    - [Step 2: add feature scripts](#step-2-add-feature-scripts)
  - [Stock Selection](#stock-selection)
    - [Step 1: add file](#step-1-add-file-1)
    - [Step 2: add selection scripts](#step-2-add-selection-scripts)


## Long/Short Strategies

### Step 1: add files
Add ```long.py``` and ```short.py``` to ```./trader/strategies/``` to represent long/short strategy modules, respectively.

```lua
.
  |-- trader
    |-- strategies
      |-- long.py  <--
      |-- short.py <--
```

### Step 2: add long strategy scripts
The script should be an class object, see the [sample code](../../docs/script%20samples/long.py) for more details.

### Step 3: add short strategy scripts
The script should be an class object, see the [sample code](../../docs/script%20samples/short.py) for more details.

<u>Be sure to update self.STRATEGIES, self.Funcs, and self.QuantityFunc (if exists) before running AutoTradingPlatform</u>

## K-Bar Features
After developing your trading strategies, you should also develop a feature script file to generate features (or technical indicators) for each strategy. Otherwise, the strategies would run with errors (lack of features). Similar to [Long/Short Strategies](#longshort-strategies), feature scripts is added by 2 steps:

### Step 1: add file
Add ```features.py``` to ```./trader/strategies/```.

```lua
.
  |-- trader
    |-- strategies
      |-- features.py <--
```

### Step 2: add feature scripts
The script should be an class object, see the [sample code](../../docs/script%20samples/features.py) for more details.


## Stock Selection
Stock selection is also an important part of auto-trading, since traders cannot trade without target securities. You can also define your own scripts only by 2 steps:

### Step 1: add file
Add ```conditions.py``` to ```./trader/strategies/```.

```lua
.
  |-- trader
    |-- strategies
      |-- conditions.py <--
```

### Step 2: add selection scripts
The script should be an class object, see the [sample code](../../docs/script%20samples/conditions.py) for more details.
