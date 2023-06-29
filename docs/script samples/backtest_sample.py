import pandas as pd
from datetime import timedelta
from collections import namedtuple

from trader.config import TODAY_STR
from trader.performance.backtest import BackTester, merge_pc_ratio
from trader.utils.select import SelectStock


def kbar_1D_processor(df1d, df):
    df1d = df1d.rename(columns={
        'OTCopen': 'OTCopenD',
        'OTChigh': 'OTChighD',
        'OTClow': 'OTClowD',
        'OTCclose': 'OTCcloseD',
        'Open': 'OpenD',
        'High': 'HighD',
        'Low': 'LowD',
        'Close': 'CloseD',
        'Volume': 'VolumeD',
        'isIn': 'isInD',
        'bias': 'biasD'
    })
    commons = list(set(df.columns).intersection(set(df1d.columns)))
    target_columns_1D = ['name'] + \
        [c for c in df1d.columns if c not in commons]
    df1d = df1d[target_columns_1D]
    return df1d


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

    # Kbar scale(also is the main scale for multiple kbar scales backtesting)
    scale = '1D'

    # whether raise trading quota after opening a position
    raiseQuota = False

    # trading leverage
    leverage = 1

    # base number of days for generating features
    kbar_start_day = 16

    # target columns for backtesting
    target_columns = [
        # required columns
        'name', 'date', 'isIn', 'nth_bar',
        'Open', 'High', 'Low', 'Close', 'Volume', 'volume_ma',
        'TSEopen', 'TSEhigh', 'TSElow', 'TSEclose', 'TSEvolume',
        'OTCopen', 'OTChigh', 'OTClow', 'OTCclose', 'OTCvolume',

        # optional columns
        'pc_ratio', 'z_totalamount', 'OTCclose_ma5',
        'yOpen', 'yHigh', 'yLow', 'yClose',
    ]

    def add_features_1d(self, df: pd.DataFrame):
        '''
        ===================================================================

        *****OPTIONAL*****

        Function of adding "Day-frequency" features. Add this function if
        your backtest strategy needs multi-scale kbar datasets.
        ===================================================================
        '''
        df['yClose'] = df.groupby('name').Close.transform('shift')
        return df

    def preprocess_df1d(self, df: pd.DataFrame):
        '''
        ===================================================================

        *****OPTIONAL*****

        Function of preprocessing Day-frequency dataset. Add this function
        if your backtest strategy needs multi-scale kbar datasets.
        ===================================================================
        '''
        selector = SelectStock(dma=5, mode=self.mode, scale='1D')
        df1d = selector.load_and_merge()
        df1d = selector.preprocess(df1d)
        df1d = self.add_features_1d(df1d)

        # label to determine whether to open a position
        df1d['isIn'] = True
        df1d['isIn'] = df1d.groupby('name').isIn.shift(1).fillna(False)
        return kbar_1D_processor(df1d, df)

    def addFeatures(self, df: pd.DataFrame):
        '''
        ===================================================================
        Functions of adding "other-frequency" features.
        ===================================================================
        '''
        return df

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
        open_condition = inputs['Close'] > inputs['Open']
        if open_condition:
            return Action(100, 'reason', 'reason', inputs['Close'])
        return Action()

    def examineClose(self, stocks: dict, inputs: dict, **kwargs):
        '''
        ===================================================================
        Set conditions to close a position.
        ===================================================================
        '''
        sell_condition = inputs['Close'] < inputs['Open']
        if sell_condition:
            return Action(100, 'reason', 'reason', inputs['Close'])
        return Action()


days = 180
init_position = 1500000
startDate = pd.to_datetime(TODAY_STR) - timedelta(days=days)

backtestScript = SampleScript()
tester = BackTester(config=backtestScript)
tester.set_scripts(backtestScript)

# Load & Merge datasets
df = tester.load_data(backtestScript)
if backtestScript.scale != '1D':
    if backtestScript.market == 'Stocks':
        df1d = backtestScript.preprocess_df1d(df)

        df['date'] = pd.to_datetime(df.Time.dt.date)
        df = df.merge(df1d, how='left', on=['name', 'date'])
        df = merge_pc_ratio(df)
        del df1d


# Add backtest features and select stocks
df = tester.addFeatures(df)
df = tester.selectStocks(df)
print('Loading data done.')


# Run backtest
TestResult = tester.run(
    df,
    startDate=startDate,
    endDate=None,
    init_position=init_position,
    unit=init_position/100000,
    target_columns=backtestScript.target_columns,
    buyOrder='Close',
    raiseQuota=False
)

if TestResult.Summary is not None:
    print(TestResult.Summary.iloc[[9, 15, 16, 17, 18, 19]])

# tester.generate_tb_reasons(TestResult.Statement)

# Plot figures
fig = tester.plot_backtest_result(TestResult)

# Output backtest results
tester.save_result(TestResult)
tester.save_figure(fig, 'Backtest Result')
