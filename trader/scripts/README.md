# Script Design Instructions

For traders who use this trading framework, they only have to create 1 script for trading, which contains the following parts:
1. Strategy settings
2. Adding featrues for trading
3. Adding featrues for stock selection
4. Calculate the quantity for open, raise, close positions
5. Conditions for checking Open/Close Positions


## Where to put the script file 
For each trading strategy, the script settings are located in ```/trader/scripts/StrategySet```. For example:

```lua
.
  |-- trader
    |-- scripts
      |-- StrategySet
        |-- Strategy1.py
        |-- Strategy2.py
        |-- Strategy3.py
        |-- ...
```

### How to create a strategy script
The script should be an class object, see the [strategy_sample.py](../../docs/script%20samples/strategy_sample.py) for more details.


## How to start the trading strategy
1. For user envs: Update the values of STRATEGY_STOCK, STRATEGY_FUTURES
2. For system configs: Update update the [STRATEGY] section in the ```config.ini```
3. If there's a stock selection condition for the strategy, update the [SELECT] section in the ```config.ini```

