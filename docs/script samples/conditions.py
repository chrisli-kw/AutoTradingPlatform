import pandas as pd


class SelectConditions:
    '''
    ===================================================================
    This module is created for stock selection. The function name format 
    is condition_{your_strategy_name} where "your_strategy_name" is the 
    same as your strategy name defined in StrategySet.py. The 
    SelectStock module will automatically set the preprocess and 
    condition functions when the program initializes.
    ===================================================================
    '''

    def condition_strategy1(self, df: pd.DataFrame, *args):
        '''
        ===================================================================
        Functions of selection conditions for the strategy.
        ===================================================================
        '''
        b1 = df.Close/df.yClose > 1.05
        b2 = df.Volume > 10000
        b3 = df.Close > df.Open
        return b1 & b2 & b3
