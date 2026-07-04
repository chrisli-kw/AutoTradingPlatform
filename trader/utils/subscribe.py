import numpy as np
import pandas as pd
from typing import Union
from threading import Lock

from . import get_contract
from .kbar import KBarTool
from .objects.data import TradeData
from ..config import API


class Subscriber(KBarTool):
    def __init__(self):
        KBarTool.__init__(self)
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

        if hasattr(tick, 'to_dict') and callable(tick.to_dict):
            tick_data = tick.to_dict()
        else:
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

        if code not in TradeData.Quotes.AllTargets:
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

        API.subscribe(API.Contracts.Indexs.TSE.TSE001, quote_type='tick')
        API.subscribe(API.Contracts.Indexs.OTC.OTC101, quote_type='tick')
        self._set_index_quote_default()

    def unsubscribe_index(self):
        '''取消訂閱指數盤中資訊'''

        API.unsubscribe(
            API.Contracts.Indexs.TSE.TSE001, quote_type='tick')
        API.unsubscribe(
            API.Contracts.Indexs.OTC.OTC101, quote_type='tick')

    def subscribe_targets(self, targets: list, quote_type: str = 'tick'):
        '''訂閱股票/期貨盤中資訊'''

        for t in targets:
            target = get_contract(t)
            API.subscribe(target, quote_type=quote_type, version='v1')

    def unsubscribe_targets(self, targets: str, quote_type: str = 'tick'):
        '''取消訂閱股票盤中資訊'''

        for t in targets:
            target = get_contract(t)
            API.unsubscribe(target, quote_type=quote_type, version='v1')

    def subscribe_all(self, targetLists: Union[list, np.array], pass_index=False):
        '''訂閱指數、tick、bidask資料'''

        self._set_target_quote_default(targetLists)
        if not pass_index:
            self.subscribe_index()
        self.subscribe_targets(targetLists, 'tick')
        self.subscribe_targets(targetLists, 'bidask')

    def unsubscribe_all(self, targetLists: Union[list, np.array]):
        '''取消訂閱指數、tick、bidask資料'''

        self.unsubscribe_index()
        self.unsubscribe_targets(targetLists, 'tick')
        self.unsubscribe_targets(targetLists, 'bidask')

    @staticmethod
    def _normalize_snapshot_targets(targets):
        if isinstance(targets, (str, dict)):
            return [targets]
        if not isinstance(targets, (list, tuple, set, np.ndarray, pd.Series)):
            return [targets]
        return list(targets)

    @staticmethod
    def _snapshot_to_dict(snapshot):
        if hasattr(snapshot, 'dict') and callable(snapshot.dict):
            return snapshot.dict()
        if hasattr(snapshot, 'to_dict') and callable(snapshot.to_dict):
            return snapshot.to_dict()
        if hasattr(snapshot, '_asdict') and callable(snapshot._asdict):
            return snapshot._asdict()
        return dict(snapshot)

    def _snapshot_contract(self, target):
        if not isinstance(target, dict):
            return get_contract(target)

        contract = target.get('contract')
        if contract is not None:
            return contract

        target_expiration = target.get('expiration')
        target_strike = target.get('strike')
        target_option_type = target.get('option_type')
        target_underlying = target.get('underlying', 'TX')
        if (
            target_expiration is not None or
            target_strike is not None or
            target_option_type is not None
        ):
            return get_contract(
                expiration=target_expiration,
                strike=target_strike,
                option_type=target_option_type,
                underlying=target_underlying
            )

        target_code = (
            target.get('target') or
            target.get('symbol') or
            target.get('code')
        )
        return get_contract(target_code)

    def snapshot_targets(
            self,
            targets: Union[str, dict, list, np.array],
            as_df: bool = True
    ):
        '''取得股票/期貨盤中資訊快照'''

        contracts = [
            self._snapshot_contract(t)
            for t in self._normalize_snapshot_targets(targets)
        ]
        snapshots = API.snapshots(contracts)

        data = [self._snapshot_to_dict(s) for s in snapshots]
        if not as_df:
            return data

        df = pd.DataFrame(data)
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        return df
