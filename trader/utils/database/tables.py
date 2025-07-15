from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import TIMESTAMP
from sqlalchemy import Column, Integer, FLOAT, String, BigInteger

from .sql import Base


class UserSettings(Base):
    __tablename__ = 'user_settings'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    account = Column(String(50), nullable=False, comment='帳戶代號')
    api_key = Column(String(200), nullable=False, comment='API 金鑰')
    secret_key = Column(String(200), nullable=False, comment='API 密鑰')
    ca_passwd = Column(String(200), comment='交易憑證密碼')
    mode = Column(
        String(10), default='Simulation', comment='交易模式 (Simulation/All)')
    init_balance = Column(Integer, default=0, comment='帳戶起始資金')
    marging_trading_amount = Column(Integer, default=0, comment='融資額度')
    short_selling_amount = Column(Integer, default=0, comment='融券額度')
    trading_period = Column(
        String(10), default='Day', comment='交易時段(Day/Night/Both)')
    margin_amount = Column(Integer, default=0, comment='可下單的保證金額上限')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class SecurityInfo(Base):
    __tablename__ = 'security_info'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    account = Column(String(50), nullable=False, comment='帳戶代號')
    market = Column(String(10), nullable=False, comment='市場別')
    code = Column(String(10), nullable=False, comment='證券代號')
    action = Column(String(10), nullable=False, comment='買賣別')
    quantity = Column(Integer, nullable=False, comment='今日庫存量')
    cost_price = Column(FLOAT(2), nullable=False, comment='成本價')
    last_price = Column(FLOAT(2), nullable=False, comment='前一日收盤價')
    pnl = Column(Integer, comment='未實現損益')
    yd_quantity = Column(Integer, comment='昨日庫存量')
    order_cond = Column(String(50), nullable=False, comment='委託類型')
    timestamp = Column(
        TIMESTAMP(fsp=6), server_default=func.now(), comment='進場時間')
    position = Column(Integer, nullable=False, comment='剩餘部位%')
    strategy = Column(String(50), server_default='unknown', comment='策略名稱')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class PositionTable(Base):
    __tablename__ = 'position_table'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    account = Column(String(50), nullable=False, comment='帳戶代號')
    strategy = Column(String(50), nullable=False, comment='策略名稱')
    name = Column(String(10), nullable=False, comment='證券名稱')
    price = Column(FLOAT(2), nullable=False, comment='進場價格')
    timestamp = Column(
        TIMESTAMP(fsp=6), server_default=func.now(), comment='進場時間')
    quantity = Column(Integer, nullable=False, comment='進場數量')
    reason = Column(String(50), nullable=False, comment='進場原因')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class SecurityList(Base):
    __tablename__ = 'security_list'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    underlying_kind = Column(String(50))
    update_date = Column(TIMESTAMP(fsp=6), server_default=func.now())
    target_code = Column(String(10))
    reference = Column(FLOAT(2), nullable=False, default=0)
    delivery_date = Column(String(10))
    exchange = Column(String(3))
    delivery_month = Column(String(6))
    name = Column(String(50), default='unknown')
    short_selling_balance = Column(Integer)
    option_right = Column(String(50))
    strike_price = Column(FLOAT(2))
    underlying_code = Column(String(10))
    margin_trading_balance = Column(Integer)
    limit_up = Column(FLOAT(2), nullable=False)
    limit_down = Column(FLOAT(2), nullable=False)
    symbol = Column(String(10))
    category = Column(String(3))
    multiplier = Column(FLOAT(2), default=0)
    currency = Column(String(3), default='TWD')
    day_trade = Column(String(7), default='No')
    code = Column(String(10), nullable=False)
    unit = Column(Integer)
    security_type = Column(String(3))

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData1T(Base):
    __tablename__ = 'kbar_data_1T'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10), nullable=False)
    Time = Column(TIMESTAMP(fsp=6), server_default=func.now())
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData30T(Base):
    __tablename__ = 'kbar_data_30T'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10), nullable=False)
    Time = Column(TIMESTAMP(fsp=6), server_default=func.now())
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData60T(Base):
    __tablename__ = 'kbar_data_60T'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10), nullable=False)
    Time = Column(TIMESTAMP(fsp=6), server_default=func.now())
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData1D(Base):
    __tablename__ = 'kbar_data_1D'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10), nullable=False)
    Time = Column(TIMESTAMP(fsp=6), server_default=func.now())
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class PutCallRatioList(Base):
    __tablename__ = 'put_call_ratio'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    Date = Column(TIMESTAMP(fsp=6), server_default=func.now(), comment='日期')
    PutVolume = Column(FLOAT(2), default=0, comment='賣權成交量')
    CallVolume = Column(FLOAT(2), default=0, comment='買權成交量')
    PutCallVolumeRatio = Column(FLOAT(2), default=0, comment='買賣權成交量比率%')
    PutOpenInterest = Column(FLOAT(2), default=0, comment='賣權未平倉')
    CallOpenInterest = Column(FLOAT(2), default=0, comment='買權未平倉')
    PutCallRatio = Column(FLOAT(2), default=0, comment='買賣權未平倉量比率%')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class ExDividendTable(Base):
    __tablename__ = 'ex_dividend_table'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    Date = Column(
        TIMESTAMP(fsp=6), server_default=func.now(), comment='除權除息日期')
    Code = Column(String(10), nullable=False, comment='股票代號')
    Name = Column(String(50), default='unknown', comment='名稱')
    DividendType = Column(String(5), comment='除權息')
    DividendRate = Column(FLOAT(2), default=0, comment='無償配股率')
    CashCapitalRate = Column(FLOAT(2), default=0, comment='現金增資配股率')
    CashCapitalPrice = Column(FLOAT(2), default=0, comment='現金增資認購價')
    CashDividend = Column(FLOAT(2), default=0, comment='現金股利')
    Details = Column(String(10), comment='詳細資料')
    Reference = Column(String(10), comment='參考價試算')
    Quarter = Column(String(10), comment='最近一次申報資料 季別/日期')
    NetValue = Column(FLOAT(2), default=0, comment='最近一次申報每股 (單位)淨值')
    EPS = Column(FLOAT(2), default=0, comment='最近一次申報每股 (單位)盈餘')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class TradingStatement(Base):
    __tablename__ = 'trading_statement'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    Time = Column(
        TIMESTAMP(fsp=6), server_default=func.now(), comment='交易日期')
    market = Column(String(10), nullable=False, comment='市場別')
    account_id = Column(
        String(50), default='unknown', comment='帳號別')
    code = Column(String(10), nullable=False, comment='證券代號')
    action = Column(String(50), default='unknown', comment='買賣別')
    price = Column(FLOAT(2), default=0, comment='成交價')
    quantity = Column(FLOAT(2), default=0, comment='成交數量')
    amount = Column(FLOAT(2), default=0, comment='成交金額')
    order_cond = Column(
        String(50), default='unknown', comment='現股或融資')
    order_lot = Column(
        String(50), default='unknown', comment='整張或零股')
    op_type = Column(
        String(50), default='unknown', comment='期權委託類型')
    leverage = Column(FLOAT(2), default=1, comment='槓桿比例')
    msg = Column(
        String(200), default='unknown', comment='進出場訊息')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class SelectedStocks(Base):
    __tablename__ = 'selected_stocks'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    code = Column(String(10), nullable=False, comment='證券代號')
    company_name = Column(String(10), comment='證券名稱')
    category = Column(String(50), comment='產業類別')
    Time = Column(
        TIMESTAMP(fsp=6), server_default=func.now(), comment='選股日期')
    Open = Column(FLOAT(2), nullable=False, comment='開盤價')
    High = Column(FLOAT(2), nullable=False, comment='最高價')
    Low = Column(FLOAT(2), nullable=False, comment='最低價')
    Close = Column(FLOAT(2), nullable=False, comment='收盤價')
    Volume = Column(Integer, comment='成交量')
    Amount = Column(BigInteger, comment='成交額')
    Strategy = Column(String(10), nullable=False, comment='選股策略')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=func.now())

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
