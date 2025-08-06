import logging
import pandas as pd
from datetime import datetime

from trader.config import TODAY_STR
from trader.utils.objects import Action
from trader.utils.objects.data import TradeData
from trader.utils.positions import Position


def dats_source(start='', end=''):
    if not start:
        start = '2000-01-01'

    if not end:
        end = TODAY_STR

    df = pd.read_csv('your_data_path')
    df = df[(df.Time >= start) & (df.Time <= end)]
    return df


class StrategyConfig:
    '''
    ===================================================================
    This is a sample of formating a backtest script.
    The following attributes are required:
    ===================================================================
    '''

    # strategy name
    name = 'LongStrategy'

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

    # Quantity for opening a position at a time
    open_qty = 1

    # Quantity for raising a position at a time
    raise_qty = 1

    # Quantity for stop loss a position at a time
    stop_loss_qty = 1

    # Quantity for stop profit a position at a time
    stop_profit_qty = 1

    # The maximum quantity of the strategy can make
    max_qty = 10

    # The price limit of a single targets, any target's price higher than the
    # threshold will be filtered out for trading monitor
    PRICE_THRESHOLD = 99999

    # The targets you want to filter OUT for trading monitor
    FILTER_OUT = []

    # The targets you want to filter IN for trading monitor
    Targets = []

    # The maximum amount can be used for MarginTrading/ShortSelling
    Margin_Trading = 0
    SHORT_SELLING = 0

    # The stock order condigion (Cash/MarginTrading/ShortSelling)
    ORDER_COND1 = 'MarginTrading'
    ORDER_COND2 = 'ShortSelling'

    # The maximum number of targets hold in a position
    PositionLimit = 0

    # True = This is a day-trade strategy .
    DayTrade = False

    positions = Position()

    # Optional: add this attribute if your strategy needs more datssets.
    extraData = dict(dats_source=dats_source)

    def __init__(self, account_name: str, strategy: str, **kwargs):
        self.account_name = account_name
        self.positions = Position(account_name=account_name, strategy=strategy)

    def add_features(self):
        '''
        ===================================================================
        Functions of adding all features.
        ===================================================================
        '''
        for scale in self.kbarScales:
            func = self.addFeatures_1D if scale == '1D' else self.addFeatures_T
            TradeData.KBars.Freq[scale] = func(TradeData.KBars.Freq[scale])

    def select_preprocess(self, df: pd.DataFrame, **kwargs):
        '''
        ===================================================================
        Like add_features(), this function generates features for selection.
        ===================================================================
        '''
        return df

    def select_condition(self, df: pd.DataFrame, **kwargs):
        '''
        ===================================================================
        Set conditions to select stocks.
        ===================================================================
        '''
        return df

    def selectStocks(self, *args):
        '''
        ===================================================================
        Set conditions to select stocks.
        ===================================================================
        '''

        df = TradeData.KBars.Freq[self.scale].copy()
        df = self.select_preprocess(df)
        df['isIn'] = self.select_condition(df)
        df.isIn = df.groupby('name').isIn.shift(1).fillna(False)
        TradeData.KBars.Freq[self.scale] = df

        if len(self.kbarScales) > 1:
            df = TradeData.KBars.Freq[self.scale]
            isIn = TradeData.KBars.Freq['1D'][['date', 'name', 'isIn']]
            df = df.merge(isIn, how='left', on=['date', 'name'])
            TradeData.KBars.Freq[self.scale] = df

    def Quantity(self, inputs: dict, **kwargs):
        '''
        ===================================================================
        Calculate the quantity & maximum amount to open/raise position.
        ===================================================================
        '''

        raise_pos = kwargs.get('raise_pos', False)
        quantity = self.raise_qty if raise_pos else self.open_qty
        target = kwargs.get('target', '')
        return quantity, self.max_qty.get(target, 0)

    @staticmethod
    def examineOpen(trade: dict, price=None) -> bool:
        '''
        ===================================================================
        Set conditions to open a position.
        ===================================================================
        '''
        open = trade.get('Open')
        return (open > 100)

    @classmethod
    def raise_position(self, trade: dict, entries: list, price=None):
        '''
        ===================================================================
        Set conditions to raise an increase in opened positions.
        ===================================================================
        '''
        if not entries:
            logging.warning('Entries list is empty')
            return False

        price = trade['tOpen'] if price is None else price
        atr = trade.get('ATR')
        return (atr > 3)

    @staticmethod
    def stop_loss(trade: dict, entries: list, price=None):
        '''
        ===================================================================
        Set conditions to stop loss.
        ===================================================================
        '''
        if not entries:
            logging.warning('Entries list is empty')
            return False

        price = trade['tOpen'] if price is None else price
        atr = trade.get('ATR')
        return (atr < 4)

    @staticmethod
    def stop_profit(trade: dict, entries: list, price=None):
        '''
        ===================================================================
        Set conditions to stop profit.
        ===================================================================
        '''
        if not entries:
            logging.warning('Entries list is empty')
            return False

        price = trade['tOpen'] if price is None else price
        atr = trade.get('ATR')
        return (atr > 10)

    def Open(self, inputs: dict, **kwargs):
        '''
        ===================================================================
        Check if currenct condition meets the opening/raising conditions.
        ===================================================================
        '''

        action = 'Buy'
        now = datetime.now()
        target = inputs['symbol']
        price = TradeData.Quotes.NowTargets[target]['price']

        K1min = TradeData.KBars.Freq[self.scale].copy()
        K1min = K1min[K1min.name == target].tail(1).to_dict('records')[0]

        # New position
        if not self.positions.entries and self.examineOpen(K1min, price=price):
            self.positions.open(price, now, qty=self.open_qty)

            reason = 'open position'
            return Action(action, reason, self.open_qty)

        # Raise position
        elif self.raise_position(K1min, self.positions, price=price):
            raise_qty = min(
                self.raise_qty, self.max_qty - self.positions.total_qty)
            self.positions.open(price, now, qty=raise_qty)

            reason = 'raise position'
            return Action(action, reason, raise_qty)

        return Action()

    def Close(self, inputs: dict, **kwargs):
        '''
        ===================================================================
        Check if currenct condition meets the stop profit/loss conditions.
        ===================================================================
        '''
        action = 'Sell'
        now = datetime.now()
        target = inputs['symbol']
        price = TradeData.Quotes.NowTargets[target]['price']

        K1min = TradeData.KBars.Freq[self.scale].copy()
        K1min = K1min[K1min.name == target].tail(1).to_dict('records')[0]
        entries = self.positions.entries

        if self.stop_loss(K1min, entries, price=price):
            self.positions.close(
                price, now, reason='stop loss', qty=self.stop_loss_qty)

            reason = 'stop loss'
            return Action(action, reason, self.stop_loss_qty)

        elif self.stop_profit(K1min, entries, price=price):
            self.positions.close(
                price, now, reason='stop profit', qty=self.stop_profit_qty)

            reason = 'stop profit'
            return Action(action, reason, self.stop_profit_qty)

        return Action()
