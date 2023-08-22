import pandas as pd

from trader.indicators.signals import TechnicalSignals


class FeaturesSelect:
    '''
    ===================================================================
    This module is created for stock selection. The function name format 
    is preprocess_{your_strategy_name} where "your_strategy_name" is the 
    same as your strategy name defined in StrategySet.py. The 
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


class KBarFeatureTool(TechnicalSignals):
    '''
    ===================================================================
    This module is created for adding KBar features. There is a name 
    format for function names: add_K{kbar_frequency}_feature, where
    "kbar_frequency" can be Day, 60min, 30min, 15min, 5min, or 1min. Each
    function is designed to add features with respect to its frequency
    data.
    ===================================================================
    '''

    def add_KDay_feature(self, KDay: pd.DataFrame):
        '''
        ===================================================================
        Functions of adding Day-frequency features.
        ===================================================================
        '''
        KDay['date'] = pd.to_datetime(KDay['date'])
        KDay['ma_1D_5'] = self._MA(KDay, 'Close', 5)
        return KDay

    def add_K60min_feature(self, K60min: pd.DataFrame):
        '''
        ===================================================================
        Functions of adding 60min-frequency features.
        ===================================================================
        '''
        K60min['ma_60T_10'] = self._MA(K60min, 'Close', 10)
        return K60min

    def add_K30min_feature(self, K30min: pd.DataFrame):
        '''
        ===================================================================
        Functions of adding 30min-frequency features.
        ===================================================================
        '''
        K30min['ma_30T_10'] = self._MA(K30min, 'Close', 10)
        return K30min
