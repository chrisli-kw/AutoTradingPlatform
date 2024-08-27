import pandas as pd


class Leverage:
    Long = {}
    Short = {}


class Stocks:
    InfoDefault = pd.DataFrame(
        columns=[
            'account', 'market',
            'code', 'order_cond', 'action', 'pnl',
            'cost_price', 'quantity', 'yd_quantity', 'last_price'
        ]
    )
    Info = pd.DataFrame()
    Strategy = {}
    Monitor = {}
    N_Long = 0
    N_Short = 0
    Leverage = Leverage()
    Bought = []
    Sold = []


class Futures:
    InfoDefault = pd.DataFrame(
        columns=[
            'account', 'market',
            'code', 'action', 'quantity', 'cost_price',
            'last_price', 'pnl'
        ]
    )
    Info = pd.DataFrame()
    Strategy = {}
    Monitor = {}
    Opened = []
    Closed = []
    Transferred = {}
    CodeList = {}


class Quotes:
    AllIndex = {'TSE001': [], 'OTC101': []}
    NowIndex = {}
    AllTargets = {}
    NowTargets = {}
    TempKbars = {}


class DefaultTableMeta(type):
    _data = {
        'Stocks': Stocks,
        'Futures': Futures,
        'BidAsk': {},
        'Quotes': Quotes()
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
