from sqlalchemy import Column, Integer, FLOAT, String, text
from sqlalchemy.dialects.mysql import TIMESTAMP

from .sql import Base


class SecurityInfoStocks(Base):
    __tablename__ = 'security_info_stocks'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    account = Column(String(50, 'utf8mb4_unicode_ci'), nullable=False, comment='帳戶代號')
    market = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False, comment='市場別')
    code = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False, comment='證券代號')
    order_cond = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False, comment='委託類型')
    action = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False, comment='買賣別')
    pnl = Column(Integer, comment='未實現損益')
    cost_price = Column(FLOAT(2), nullable=False, comment='成本價')
    quantity = Column(Integer, nullable=False, comment='今日庫存量')
    yd_quantity = Column(Integer, comment='昨日庫存量')
    last_price = Column(FLOAT(2), nullable=False, comment='前一日收盤價')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class SecurityInfoFutures(Base):
    __tablename__ = 'security_info_futures'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    Account = Column(String(50, 'utf8mb4_unicode_ci'), nullable=False, comment='帳戶代號')
    Market = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False, comment='市場別')
    Date = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"), nullable=False, comment='日期')
    Code = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False, comment='證券代號')
    CodeName = Column(String(50, 'utf8mb4_unicode_ci'), comment='標的名稱')
    OrderNum = Column(String(50, 'utf8mb4_unicode_ci'), comment='委託編號')
    OrderBS = Column(String(50, 'utf8mb4_unicode_ci'), comment='買賣別')
    OrderType = Column(String(50, 'utf8mb4_unicode_ci'), comment='委託類型')
    Currency = Column(String(50, 'utf8mb4_unicode_ci'), comment='幣別')
    paddingByte = Column(String(50, 'utf8mb4_unicode_ci'), comment='unknown')
    Volume = Column(FLOAT(2), comment='成交量')
    ContractAverPrice = Column(FLOAT(2), comment='平均合約價')
    SettlePrice = Column(FLOAT(2), comment='履約價')
    RealPrice = Column(FLOAT(2), comment='unknown')
    FlowProfitLoss = Column(FLOAT(2), comment='unknown')
    SettleProfitLoss = Column(FLOAT(2), comment='unknown')
    StartSecurity = Column(String(50, 'utf8mb4_unicode_ci'), comment='unknown')
    UpKeepSecurity = Column(String(50, 'utf8mb4_unicode_ci'), comment='unknown')
    OTAMT = Column(FLOAT(2), comment='unknown')
    MTAMT = Column(FLOAT(2), comment='unknown')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class Watchlist(Base):
    __tablename__ = 'watchlist'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    account = Column(String(50, 'utf8mb4_unicode_ci'), nullable=False)
    market = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False)
    code = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False)
    buyday = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    bsh = Column(FLOAT(2), nullable=False)
    position = Column(Integer, nullable=False)
    strategy = Column(String(50, 'utf8mb4_unicode_ci'), server_default='unknown')
    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class SecurityList(Base):
    __tablename__ = 'security_list'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    underlying_kind = Column(String(50, 'utf8mb4_unicode_ci'))
    update_date = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    target_code = Column(String(10, 'utf8mb4_unicode_ci'))
    reference = Column(FLOAT(2), nullable=False)
    delivery_date = Column(String(10, 'utf8mb4_unicode_ci'))
    exchange = Column(String(3, 'utf8mb4_unicode_ci'))
    delivery_month = Column(String(6, 'utf8mb4_unicode_ci'))
    name = Column(String(50, 'utf8mb4_unicode_ci'), default='unknown')
    short_selling_balance = Column(Integer)
    option_right = Column(String(50, 'utf8mb4_unicode_ci'))
    strike_price = Column(FLOAT(2))
    underlying_code = Column(String(10, 'utf8mb4_unicode_ci'))
    margin_trading_balance = Column(Integer)
    limit_up = Column(FLOAT(2), nullable=False)
    limit_down = Column(FLOAT(2), nullable=False)
    symbol = Column(String(10, 'utf8mb4_unicode_ci'))
    category = Column(String(3, 'utf8mb4_unicode_ci'))
    multiplier = Column(FLOAT(2), default=0)
    currency = Column(String(3, 'utf8mb4_unicode_ci'), default='TWD')
    day_trade = Column(String(7, 'utf8mb4_unicode_ci'), default='No')
    code = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False)
    unit = Column(Integer)
    security_type = Column(String(3, 'utf8mb4_unicode_ci'))

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData1T(Base):
    __tablename__ = 'kbar_data_1T'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False)
    date = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    Time = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    hour = Column(Integer)
    minute = Column(Integer)
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData30T(Base):
    __tablename__ = 'kbar_data_30T'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False)
    date = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    Time = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    hour = Column(Integer)
    minute = Column(Integer)
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData60T(Base):
    __tablename__ = 'kbar_data_60T'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False)
    date = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    Time = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    hour = Column(Integer)
    minute = Column(Integer)
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData1D(Base):
    __tablename__ = 'kbar_data_1D'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False)
    date = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    Time = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    hour = Column(Integer)
    minute = Column(Integer)
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class PutCallRatioList(Base):
    __tablename__ = 'put_call_ratio'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    Date = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"), comment='日期')
    PutVolume = Column(FLOAT(2), default=0, comment='賣權成交量')
    CallVolume = Column(FLOAT(2), default=0, comment='買權成交量')
    PutCallVolumeRatio = Column(FLOAT(2), default=0, comment='買賣權成交量比率%')
    PutOpenInterest = Column(FLOAT(2), default=0, comment='賣權未平倉')
    CallOpenInterest = Column(FLOAT(2), default=0, comment='買權未平倉')
    PutCallRatio = Column(FLOAT(2), default=0, comment='買賣權未平倉量比率%')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class ExDividendTable(Base):
    __tablename__ = 'ex_dividend_table'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    Date = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"), comment='除權除息日期')
    Code = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False, comment='股票代號')
    Name = Column(String(50, 'utf8mb4_unicode_ci'), default='unknown', comment='名稱')
    DividendType = Column(String(5, 'utf8mb4_unicode_ci'), comment='除權息')
    DividendRate = Column(FLOAT(2), default=0, comment='無償配股率')
    CashCapitalRate = Column(FLOAT(2), default=0, comment='現金增資配股率')
    CashCapitalPrice = Column(FLOAT(2), default=0, comment='現金增資認購價')
    CashDividend = Column(FLOAT(2), default=0, comment='現金股利')
    Details = Column(String(10, 'utf8mb4_unicode_ci'), comment='詳細資料')
    Reference = Column(String(10, 'utf8mb4_unicode_ci'), comment='參考價試算')
    Quarter = Column(String(10, 'utf8mb4_unicode_ci'), comment='最近一次申報資料 季別/日期')
    NetValue = Column(FLOAT(2), default=0, comment='最近一次申報每股 (單位)淨值')
    EPS = Column(FLOAT(2), default=0, comment='最近一次申報每股 (單位)盈餘')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class TradingStatement(Base):
    __tablename__ = 'trading_statement'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    Time = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"), comment='除權除息日期')
    market = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False, comment='市場別')
    account_id = Column(String(50, 'utf8mb4_unicode_ci'), default='unknown', comment='帳號別')
    code = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False, comment='證券代號')
    action = Column(String(50, 'utf8mb4_unicode_ci'), default='unknown', comment='買賣別')
    price = Column(FLOAT(2), default=0, comment='成交價')
    quantity = Column(FLOAT(2), default=0, comment='成交數量')
    amount = Column(FLOAT(2), default=0, comment='成交金額')
    order_cond = Column(String(50, 'utf8mb4_unicode_ci'), default='unknown', comment='現股或融資')
    order_lot = Column(String(50, 'utf8mb4_unicode_ci'), default='unknown', comment='整張或零股')
    op_type = Column(String(50, 'utf8mb4_unicode_ci'), default='unknown', comment='期權委託類型')
    leverage = Column(FLOAT(2), default=1, comment='槓桿比例')
    msg = Column(String(200, 'utf8mb4_unicode_ci'), default='unknown', comment='進出場訊息')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)