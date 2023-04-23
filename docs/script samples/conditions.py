import pandas as pd


class SelectConditions:
    '''
    ===================================================================
    This module is created for stock selection. There is 2 name format 
    for the 2 important steps: 1. preprocess_{your_strategy_name} and 
    2. condition_{your_strategy_name} where your_strategy_name is the 
    same as your strategy name defined in long/short.py. The 
    SelectStock module will automatically set the preprocess and 
    condition functions when the program initializes.
    ===================================================================
    '''

    def preprocess_strategy1(self, df: pd.DataFrame):
        '''
        ===================================================================
        Functions of preprocessing data for the strategy.
        ===================================================================
        '''
        group = df.groupby('name')
        df['yClose'] = group.Close.transform('shift')
        return df

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
