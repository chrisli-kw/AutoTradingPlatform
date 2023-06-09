import numpy as np
from typing import Union

from .. import API
from . import get_contract


class Quotes:
    AllIndex = {'TSE001': [], 'OTC101': []}
    NowIndex = {}
    AllTargets = {}
    NowTargets = {}


class Subscriber:
    def __init__(self):
        # 即時成交資料, 所有成交資料, 下單資料
        self.BidAsk = {}
        self.Quotes = Quotes()

    def _set_target_quote_default(self, targets: str):
        '''初始化股票/期權盤中資訊'''
        keys = ['price', 'amount', 'total_amount', 'volume', 'total_volume', 'tick_type']
        self.Quotes.AllTargets = {s: {k: [] for k in keys} for s in targets}
    
    def _set_index_quote_default(self):
        '''初始化指數盤中資訊'''
        self.Quotes.AllIndex = {'TSE001': [], 'OTC101': []}

    def index_v0(self, quote: dict):
        if quote['Code'] == '001':
            self.Quotes.NowIndex['TSE001'] = quote
            self.Quotes.AllIndex['TSE001'].append(quote)
        elif quote['Code'] == '101':
            self.Quotes.NowIndex['OTC101'] = quote
            self.Quotes.AllIndex['OTC101'].append(quote)

    def stk_quote_v1(self, tick):
        '''處理股票即時成交資料'''
        tick_data = dict(tick)
        for k in [
            'open', 'close', 'high', 'low', 'amount', 'total_amount', 'total_volume',
            'avg_price', 'price_chg', 'pct_chg'
        ]:
            tick_data[k] = float(tick_data[k])
        tick_data['price'] = tick_data['close']

        for k in ['price', 'amount', 'total_amount', 'volume', 'total_volume', 'tick_type']:
            self.Quotes.AllTargets[tick.code][k].append(tick_data[k])

        self.Quotes.NowTargets[tick.code] = tick_data
        return tick_data

    def fop_quote_v1(self, symbol: str, tick):
        '''處理期權即時成交資料'''
        tick_data = dict(tick)
        for k in [
            'open', 'close', 'high', 'low', 'amount', 'total_amount',
            'underlying_price', 'avg_price', 'price_chg', 'pct_chg'
        ]:
            tick_data[k] = float(tick_data[k])

        tick_data['price'] = tick_data['close']
        tick_data['symbol'] = symbol

        for k in ['price', 'amount', 'total_amount', 'volume', 'total_volume', 'tick_type']:
            self.Quotes.AllTargets[symbol][k].append(tick_data[k])

        self.Quotes.NowTargets[symbol] = tick_data
        return tick_data

    def subscribe_index(self):
        '''訂閱指數盤中資訊'''

        API.quote.subscribe(API.Contracts.Indexs.TSE.TSE001, quote_type='tick')
        API.quote.subscribe(API.Contracts.Indexs.OTC.OTC101, quote_type='tick')
        self._set_index_quote_default()

    def unsubscribe_index(self):
        '''取消訂閱指數盤中資訊'''

        API.quote.unsubscribe(API.Contracts.Indexs.TSE.TSE001, quote_type='tick')
        API.quote.unsubscribe(API.Contracts.Indexs.OTC.OTC101, quote_type='tick')

    def subscribe_targets(self, targets: list, quote_type: str = 'tick'):
        '''訂閱股票/期貨盤中資訊'''

        for t in targets:
            target = get_contract(t)
            API.quote.subscribe(target, quote_type=quote_type, version='v1')

    def unsubscribe_targets(self, targets: str, quote_type: str = 'tick'):
        '''取消訂閱股票盤中資訊'''

        for t in targets:
            target = get_contract(t)
            API.quote.unsubscribe(target, quote_type=quote_type, version='v1')

    def subscribe_all(self, targetLists: Union[list, np.array]):
        '''訂閱指數、tick、bidask資料'''

        self.subscribe_index()
        self.subscribe_targets(targetLists, 'tick')
        self.subscribe_targets(targetLists, 'bidask')
        self._set_target_quote_default(targetLists)

    def unsubscribe_all(self, targetLists: Union[list, np.array]):
        '''取消訂閱指數、tick、bidask資料'''

        self.unsubscribe_index()
        self.unsubscribe_targets(targetLists, 'tick')
        self.unsubscribe_targets(targetLists, 'bidask')
