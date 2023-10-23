import numpy as np
import pandas as pd

from ..config import StrategyList


def AccountingNumber(number: float):
    return f"{'{:,}'.format(number)}"


def convert_statement(df, mode='trading', **kwargs):
    def extract_strategy(msg):
        for s in StrategyList.All:
            if s in msg:
                return s
        return 'Unknown'

    if not df.shape[0]:
        return df

    init_position = kwargs.get('init_position', 1000000)
    if mode == 'trading':
        df = df.rename(columns={
            'code': 'Code',
            'price': 'Price',
            'quantity': 'Quantity',
            'msg': 'Reason',
            'amount': 'Amount'
        })
        df['Strategy'] = df.Reason.apply(extract_strategy)
        df['isLong'] = df.Strategy.apply(lambda x: x in StrategyList.Long)
        df['isShort'] = df.isLong == False
        df['isOpen'] = (
            ((df.isLong == True) & (df.action == 'Buy')) |
            ((df.isShort == True) & (df.action == 'Sell'))
        )

        target_columns1 = ['Strategy', 'Code', 'isLong', 'isShort']
        target_columns2 = ['Time', 'Price', 'Quantity', 'Amount', 'Reason']
        tb1 = df[df.isOpen == True][target_columns1 + target_columns2]
        tb1.columns = target_columns1 + ['Open' + c for c in target_columns2]

        tb2 = df[df.isOpen == False][target_columns1 + target_columns2]
        tb2.columns = target_columns1 + ['Close' + c for c in target_columns2]

        tb = pd.concat([tb1, tb2]).sort_index()
        tb.CloseAmount = tb.CloseAmount.abs()
        tb.OpenQuantity = tb.OpenQuantity.fillna(tb.CloseQuantity)
        for c in ['Time', 'Price', 'Amount', 'Reason']:
            col = f'Open{c}'
            tb[col] = tb.groupby('Code')[col].ffill()

        tb['OpenAmount'] = (tb.OpenPrice*tb.OpenQuantity).abs()
        tb['CloseAmount'] = (tb.ClosePrice*tb.CloseQuantity).abs()
        tb['profit'] = (tb.CloseAmount - tb.OpenAmount)*(tb.isShort*(-2)+1)
        tb['returns'] = 100*(tb.profit/tb.OpenAmount).round(4)
        tb['balance'] = init_position + tb.profit.cumsum()
        tb = tb.dropna().reset_index(drop=True)

        for index, row in tb.iterrows():
            sub_string = f'【{row.Code}】{row.Strategy}'
            tb.at[index, 'OpenReason'] = row.OpenReason.replace(
                sub_string, '').replace('-', '')
            tb.at[index, 'CloseReason'] = row.CloseReason.replace(
                sub_string, '').replace('-', '')
        return tb

    else:
        df.OpenAmount = df.OpenAmount.astype('int64')
        df.CloseAmount = df.CloseAmount.astype('int64')
        df.ClosePrice = df.ClosePrice.round(2)

        if kwargs['market'] == 'Stocks':
            netOpenAmount = (df.OpenAmount + df.OpenFee)
            netCloseAmount = (df.CloseAmount - df.CloseFee - df.Tax)
            df['profit'] = (netCloseAmount - netOpenAmount).astype('int64')
            df['returns'] = (
                100*(df.CloseAmount/df.OpenAmount - 1)).round(2)
        else:
            sign = 1 if kwargs['isLong'] else -1

            df['profit'] = (df.ClosePrice - df.OpenPrice)*df.CloseQuantity
            totalExpense = (df.OpenFee + df.CloseFee + df.Tax)*sign
            df.profit = df.profit*kwargs['multipler'] - totalExpense
            df['returns'] = (
                sign*100*((df.ClosePrice/df.OpenPrice)**sign - 1)).round(2)

        df.profit = df.profit.round()
        df['iswin'] = df.profit > 0

        if not kwargs['isLong']:
            df.profit *= -1
            df.returns *= -1
        df['balance'] = init_position + df.profit.cumsum()

        return df


def compute_profits(tb):
    total_profit = tb.profit.sum()
    df_profit = tb[tb.profit > 0]
    df_loss = tb[tb.profit <= 0]

    has_profits = df_profit.shape[0]
    has_loss = df_loss.shape[0]

    # 毛利/毛損
    gross_profit = df_profit.profit.sum() if has_profits else 0
    gross_loss = df_loss.profit.sum() if has_loss else 0
    profit_factor = round(
        abs(gross_profit/gross_loss), 2) if gross_loss else np.inf
    if total_profit < 0:
        profit_factor *= -1

    # 平均獲利/虧損金額
    mean_profit = df_profit.profit.mean() if has_profits else 0
    mean_loss = df_loss.profit.mean() if has_loss else 0
    max_profit = df_profit.profit.max() if has_profits else 0
    max_loss = df_loss.profit.min() if has_loss else 0

    ratio1 = round(abs(mean_profit/mean_loss), 4) if mean_loss else np.inf
    profits = {
        'TotalProfit': round(total_profit),
        'GrossProfit': round(gross_profit),
        'GrossLoss': round(gross_loss),
        'MeanProfit': round(mean_profit if mean_profit else 0),
        'MeanLoss': round(mean_loss if mean_loss else 0),
        'MaxProfit': max_profit,
        'MaxLoss': max_loss,
        'ProfitFactor': profit_factor,
        'ProfitRatio': ratio1
    }

    if 'KRun' in tb.columns:
        profits.update({
            'KRunProfit': round(df_profit.KRun.mean(), 1) if has_profits else 0,
            'KRunLoss': round(df_loss.KRun.mean(), 1) if has_loss else 0
        })
    else:
        profits.update({
            'KRunProfit': None,
            'KRunLoss': None
        })

    return profits


def computeReturn(df, target1, target2):
    if df.shape[0] and target1 in df.columns and target2 in df.columns:
        start = df[target1].values[0]
        end = df[target2].values[-1]
        return 100*round(end/start - 1, 2)
    return 0


def computeWinLoss(df: pd.DataFrame):
    '''Count wins and losses'''
    win_loss = (df.profit > 0).value_counts().to_dict()
    if True not in win_loss:
        win_loss[True] = 0

    if False not in win_loss:
        win_loss[False] = 0

    return win_loss
