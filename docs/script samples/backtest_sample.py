import pandas as pd
from datetime import timedelta
from collections import namedtuple

from trader.config import TODAY_STR
from trader.performance.backtest import BackTester, merge_pc_ratio
from trader.performance.reports import BacktestReport
from trader.utils.select import SelectStock


Action = namedtuple(
    typename="Action",
    field_names=['position', 'reason', 'msg', 'price'],
    defaults=[0, '', '', 0]
)


class SampleScript:
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
    7. raiseQuota
    8. leverage
    9. kbar_start_day
    10. target_columns
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

    # base number of days for generating features
    kbar_start_day = 16

    def addFeatures_1D(self, df: pd.DataFrame):
        '''
        ===================================================================

        *****OPTIONAL*****

        Function of adding "Day-frequency" features. Add this function if
        your backtest strategy needs multi-scale kbar datasets.
        ===================================================================
        '''
        df['yClose'] = df.groupby('name').Close.transform('shift')
        return df

    def addFeatures(self, Kbars: dict):
        '''
        ===================================================================
        Functions of adding "other-frequency" features.
        ===================================================================
        '''
        Kbars['1D'] = self.addFeatures_1D(Kbars['1D'])
        return Kbars

    def selectStocks(self, df: pd.DataFrame):
        '''
        ===================================================================
        Set conditions to select stocks.
        ===================================================================
        '''
        if 'isIn' in df.columns:
            df = df.drop('isIn', axis=1)

        df.isIn = df.groupby('name').isIn.shift(1).fillna(False)
        return df

    def computeStocksLimit(self, df: pd.DataFrame, **kwargs):
        return 10000

    def examineOpen(self, inputs: dict, **kwargs):
        '''
        ===================================================================
        Set conditions to open a position.
        ===================================================================
        '''
        kbar1D = inputs['1D']
        open_condition = kbar1D['Close'] > kbar1D['Open']
        if open_condition:
            return Action(100, 'reason', 'reason', kbar1D['Close'])
        return Action()

    def examineClose(self, stocks: dict, inputs: dict, **kwargs):
        '''
        ===================================================================
        Set conditions to close a position.
        ===================================================================
        '''
        kbar1D = inputs['1D']
        sell_condition = kbar1D['Close'] < kbar1D['Open']
        if sell_condition:
            return Action(100, 'reason', 'reason', kbar1D['Close'])
        return Action()


days = 180
init_position = 1500000
startDate = pd.to_datetime(TODAY_STR) - timedelta(days=days)

backtestScript = SampleScript()
br = BacktestReport(backtestScript.market)
tester = BackTester(config=backtestScript)

# Load & Merge datasets
Kbars = tester.load_datasets(
    backtestScript,
    start=startDate,
    end='',
)

# Add backtest features and select stocks
Kbars = tester.addFeatures(Kbars)
Kbars = tester.selectStocks(Kbars)

# Run backtest
TestResult = tester.run(
    Kbars,
    init_position=init_position,
    unit=init_position/100000,
    buyOrder='Close',
    raiseQuota=False
)

if TestResult.Summary is not None:
    print(TestResult.Summary)

# tester.generate_tb_reasons(TestResult.Statement)

# Plot figures
fig = br.plot_backtest_result(TestResult)

# Output backtest results
tester.save_result(TestResult)
br.save_figure(fig, 'Backtest Result')
