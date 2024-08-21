import pandas as pd


class Stocks:
    InfoDefault = pd.DataFrame(
        columns=[
            'account', 'market',
            'code', 'order_cond', 'action', 'pnl',
            'cost_price', 'quantity', 'yd_quantity', 'last_price'
        ]
    )


class Futures:
    InfoDefault = pd.DataFrame(
        columns=[
            'account', 'market',
            'code', 'action', 'quantity', 'cost_price',
            'last_price', 'pnl'
        ]
    )
    Strategy = {}
    Opened = []
    Closed = []
    Transferred = {}
    CodeList = {}


class DefaultTableMeta(type):
    _data = {
        'Stocks': Stocks,
        'Futures': Futures
    }

    def __getitem__(cls, key):
        return cls._data[key]

    def __getattr__(cls, name):
        if name in cls._data:
            return cls._data[name]
        raise AttributeError(
            f"'{cls.__name__}' object has no attribute '{name}'")


class TradeData(metaclass=DefaultTableMeta):
    pass
