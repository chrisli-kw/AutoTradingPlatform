import os
import pandas as pd
from datetime import timedelta

from trader.config import TODAY_STR
from trader.performance.backtest import BackTester
from trader.performance.reports import BacktestReport
from trader.scripts.conditions import SelectConditions
from trader.scripts.StrategySet import StrategySet
from trader.scripts.features import FeatureTrading, FeaturesSelect


def dats_source(start='', end=''):
    if not start:
        start = '2000-01-01'

    if not end:
        end = TODAY_STR

    df = pd.read_csv('your_data_path')
    df = df[(df.Time >= start) & (df.Time <= end)]
    return df


class SampleScript(StrategySet, FeaturesSelect, FeatureTrading, SelectConditions):
    '''
    ===================================================================
    This is a sample of formating a backtest script.
    The following attributes are required:
    1. strategy
    2. market
    3. margin
    4. multipler
    5. mode
    6. scale
    7. kbarScales
    8. raiseQuota
    9. leverage
    10. extraData
    ===================================================================
    '''

    # strategy name
    strategy = 'LongStrategy'

    # Stocks/Futures market
    market = 'Stocks'

    # futures trading margin
    margin = 0

    # futures trading multipler for computing profits
    multipler = 1

    # Define whether its a long or short strategy
    mode = 'long'

    # Kbar MAIN scale for backtesting
    scale = '5T'

    # Kbar scales to generate indicators for backtesting
    kbarScales = ['1D', '5T']

    # whether raise trading quota after opening a position
    raiseQuota = False

    # trading leverage
    leverage = 1

    # Optional: add this attribute if your strategy needs more datssets.
    extraData = dict(dats_source=dats_source)

    def addFeatures_1D(self, df: pd.DataFrame):
        '''
        ===================================================================

        *****OPTIONAL*****

        Function of adding "Day-frequency" features. Add this function if
        your backtest strategy needs multi-scale kbar datasets.
        ===================================================================
        '''
        df = self.preprocess_common(df)
        df = getattr(self, f'preprocess_{self.strategy}')(df)
        df = getattr(self, f'addFeatures_1D_{self.strategy}')(df, 'backtest')
        return df

    def addFeatures_T(self, df: pd.DataFrame, scale: str):
        '''
        ===================================================================
        Functions of adding "other-frequency" features.
        ===================================================================
        '''
        func = getattr(self, f'addFeatures_{scale}_{self.strategy}')
        df = func(df, 'backtest')
        return df

    def addFeatures(self, Kbars: dict):
        for scale in self.kbarScales:
            if scale == '1D':
                Kbars[scale] = self.addFeatures_1D(Kbars[scale])
            else:
                Kbars[scale] = self.addFeatures_T(Kbars[scale], scale)
        return Kbars

    def selectStocks(self, Kbars: dict, *args):
        '''
        ===================================================================
        Set conditions to select stocks.
        ===================================================================
        '''
        df = Kbars['1D']
        df['isIn'] = getattr(self, f'condition_{self.strategy}')(df, *args)
        df.isIn = df.groupby('name').isIn.shift(1).fillna(False)
        Kbars['1D'] = df

        if len(self.kbarScales) > 1:
            df = Kbars[self.scale]
            isIn = Kbars['1D'][['date', 'name', 'isIn']]
            df = df.merge(isIn, how='left', on=['date', 'name'])
            Kbars[self.scale] = df

        return Kbars

    def setVolumeProp(self, market_value):
        return self.get_volume_prop(self.strategy, market_value)

    def computeOpenLimit(self, KBars: dict, **kwargs):
        '''
        ===================================================================
        Determine the daily limit to open a position.
        ===================================================================
        '''
        if not hasattr(self, f'openLimit_{self.strategy}'):
            return 2000

        func = getattr(self, f'openLimit_{self.strategy}')
        return func(KBars, 'backtest')

    def computeOpenUnit(self, kbars: dict):
        if not hasattr(self, f'quantity_{self.strategy}'):
            return 5

        func = getattr(self, f'quantity_{self.strategy}')
        quantity, _ = func(None, kbars, 'backtest')
        return quantity

    def examineOpen(self, inputs: dict, kbars: dict, **kwargs):
        '''
        ===================================================================
        Set conditions to open a position.
        ===================================================================
        '''
        func = self.mapFunction('Open', self.tradeType, self.strategy)
        return func(inputs=inputs, kbars=kbars, mode='backtest', **kwargs)

    def examineClose(self, inputs: dict, kbars: dict, **kwargs):
        '''
        ===================================================================
        Set conditions to close a position.
        ===================================================================
        '''
        func = self.mapFunction('Close', self.tradeType, self.strategy)
        return func(inputs=inputs, kbars=kbars, mode='backtest', **kwargs)


tester = BackTester()
backtestScript = SampleScript()
tester.set_scripts(backtestScript)


# Load & Merge datasets
Kbars = tester.load_datasets(
    start=pd.to_datetime(TODAY_STR) - timedelta(days=365),
    end='',
    dataPath=f'{os.getcwd()}/data'
)

# Add backtest features and select stocks
Kbars = tester.addFeatures(Kbars)
Kbars = tester.selectStocks(Kbars)

# Run backtest
init_position = 1000000
params = dict(init_position=init_position, buyOrder='Close')
TestResult = tester.run(Kbars, **params)

if TestResult.Summary is not None:
    print(TestResult.Summary)

# Plot figures
br = BacktestReport(backtestScript)
fig = br.plot_backtest_result(TestResult)

# Output backtest results
tester.save_result(TestResult)
br.save_figure(fig, 'Backtest Result')
