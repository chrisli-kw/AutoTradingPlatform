import numpy as np
import pandas as pd
from typing import Union
from threading import Lock

from ..config import API
from . import get_contract
from .kbar import KBarTool
from .objects.data import TradeData


class Subscriber(KBarTool):
    def __init__(self, kbar_start_day=''):
        KBarTool.__init__(self, kbar_start_day)
        self.lock = Lock()

    def _set_target_quote_default(self, targets: list):
        '''初始化股票/期權盤中資訊'''

        def default_values():
            return {
                'Time': None,
                'Open': None,
                'High': -np.inf,
                'Low': np.inf,
                'Close': None,
                'Volume': 0,
                'Amount': 0
            }

        TradeData.Quotes.AllTargets.update({
            s: default_values() for s in targets
        })

    def _set_index_quote_default(self):
        '''初始化指數盤中資訊'''
        TradeData.Quotes.AllIndex = {'TSE001': [], 'OTC101': []}

    def index_v0(self, quote: dict):
        indexes = {'001': 'TSE001', '101': 'OTC101'}
        code = indexes[quote['Code']]

        if code in TradeData.Quotes.NowIndex:
            t1 = pd.to_datetime(quote['Time'])
            t2 = pd.to_datetime(TradeData.Quotes.NowIndex[code]['Time'])

            if code in TradeData.Quotes.NowIndex and t1.minute != t2.minute:
                with self.lock:
                    self._update_K1(code, quote_type='Index')
                    TradeData.Quotes.AllIndex[code] = []

        TradeData.Quotes.NowIndex[code] = quote
        TradeData.Quotes.AllIndex[code].append(quote)

    def update_quote_v1(self, tick, code=''):
        '''處理即時成交資料'''

        tick_data = dict(tick)

        if code == '':
            code = tick.code
        else:
            tick_data['symbol'] = code

        for k in [
            'open', 'high', 'low', 'close',
            'amount', 'total_amount', 'total_volume',
            'avg_price', 'price_chg', 'pct_chg', 'underlying_price'
        ]:
            if k in tick_data:
                tick_data[k] = float(tick_data[k])

        price = tick_data['close']
        tick_data['price'] = price

        if (
            code in TradeData.Quotes.NowTargets and
            tick_data['datetime'].minute != TradeData.Quotes.NowTargets[code]['datetime'].minute
        ):
            with self.lock:
                self._update_K1(code)
                self._set_target_quote_default([code])

        kbar_data = TradeData.Quotes.AllTargets[code].copy()
        TradeData.Quotes.AllTargets[code] = {
            'Open': price if kbar_data['Open'] is None else kbar_data['Open'],
            'High': max(kbar_data['High'], price),
            'Low': min(kbar_data['Low'], price),
            'Close': price,
            'Volume': kbar_data['Volume'] + tick_data['volume'],
            'Amount': kbar_data['Amount'] + tick_data['amount'],
        }

        TradeData.Quotes.NowTargets[code] = tick_data
        return tick_data

    def subscribe_index(self):
        '''訂閱指數盤中資訊'''

        API.quote.subscribe(API.Contracts.Indexs.TSE.TSE001, quote_type='tick')
        API.quote.subscribe(API.Contracts.Indexs.OTC.OTC101, quote_type='tick')
        self._set_index_quote_default()

    def unsubscribe_index(self):
        '''取消訂閱指數盤中資訊'''

        API.quote.unsubscribe(
            API.Contracts.Indexs.TSE.TSE001, quote_type='tick')
        API.quote.unsubscribe(
            API.Contracts.Indexs.OTC.OTC101, quote_type='tick')

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
