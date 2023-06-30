import numpy as np
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import file_handler
from ..config import PATH, TODAY_STR
from ..utils import progress_bar


class FigureInOut:
    pass


class BacktestFigures:
    def __init__(self, market) -> None:
        self.Figures = FigureInOut
        self.Market = market
        self.DATAPATH = f'{PATH}/backtest'
        self.stock_col_maps = {
            'TSEopen': 'first',
            'TSEhigh': 'max',
            'TSElow': 'min',
            'TSEclose': 'last',
            'TSEvolume': 'sum',
            'OTCopen': 'first',
            'OTChigh': 'max',
            'OTClow': 'min',
            'OTCclose': 'last',
            'OTCvolume': 'sum',
            'chance': 'first',
            'n_stock_limit': 'first',
            'n_stocks': 'last',
            'nClose': 'sum',
            'balance': 'last',
            'nOpen': 'sum',
            'profit': 'sum',
            'PutCallRatio': 'first'
        }
        self.futures_col_maps = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'chance': 'first',
            'n_stock_limit': 'first',
            'n_stocks': 'last',
            'nClose': 'sum',
            'balance': 'last',
            'nOpen': 'sum',
            'profit': 'sum',
            'PutCallRatio': 'first'
        }
        self.Titles = (
            # tables
            'Backtest Settings', 'Summary', "Transaction Detail",

            # candles
            "TWSE", "OTC",

            # lines/scatters
            "Put/Call Ratio",
            "Balance", "Accumulated Profit", "Accumulated Loss",
            "Changes in Opens & Closes", "Changes in Portfolio Control"
        )
        self.TraceSettings = [
            ('TSE', 'TWSE', '#2b8de3'),
            ('OTC', 'OTC', '#91b536'),
            [
                ('PutCallRatio', 'Put/Call Ratio', '#ff9f1a'),
                ('pc115', '多空分界', '#68c035'),
            ],
            ('balance', 'Accumulated Balance', '#d3503c'),
            [
                ('profits', "Accumulated Profit", '#c25656'),
                ('losses', "Accumulated Loss", '#9ad37e')
            ],
            [
                ('nOpen', 'Opens', '#48b2ef'),
                ('nClose', 'Closes', '#7F7F7F'),
                ('n_stocks', 'in-stock', '#ef488e')
            ],
            [
                ('chance', 'Opens Available', '#48efec'),
                ('n_stock_limit', 'Opens Limit', '#b1487f')
            ]
        ]

    def _replaceString(self, x: str):
        return str(x).replace(' ', '<br>').replace('00.000000000', '00')

    def _daily_info_processor(self, TestResult: object):
        profit = TestResult.Statement.groupby('CloseDate').profit.sum()
        profit = profit.to_dict()

        daily_info = TestResult.DailyInfo
        daily_info['profit'] = daily_info.index.map(profit).fillna(0).values

        col_maps = self.stock_col_maps if self.Market == 'Stocks' else self.futures_col_maps
        daily_info = daily_info.resample(
            '1D', closed='left', label='left').apply(col_maps).dropna()
        daily_info['profits'] = (
            daily_info.profit*(daily_info.profit > 0)).cumsum()
        daily_info['losses'] = (
            daily_info.profit*(daily_info.profit <= 0)).cumsum()
        daily_info['pc115'] = 115
        return daily_info

    def subplot_add_table(self, fig: make_subplots, table: pd.DataFrame, row: int, col: int, **cellargs):
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(table.columns),
                    font=dict(size=15),
                    align="center"
                ),
                cells=dict(
                    values=[table[k].tolist() for k in table.columns],
                    align="center",
                    font=dict(size=13),
                    **cellargs
                )
            ),
            row=row,
            col=col
        )
        return fig

    def _add_trace(self, fig: make_subplots, df: pd.DataFrame, row: int, *args, **kwargs):

        if self.Market == 'Stocks':
            candle = args[0] if args[0] in ['TSE', 'OTC'] else None
            open = f'{candle}open'
            high = f'{candle}high'
            low = f'{candle}low'
            close = f'{candle}close'
            name = f'{candle}volume'
        else:
            candle = None
            open = 'Open'
            high = 'High'
            low = 'Low'
            close = 'Close'
            name = 'Volume'

        if candle:
            d = 10
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df[open],
                    high=df[high],
                    low=df[low],
                    close=df[close],
                    name=candle,
                    increasing=dict(line=dict(color='#e63746')),
                    decreasing=dict(line=dict(color='#42dd31'))
                ),
                row=row,
                col=1,
                secondary_y=True
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[close].rolling(d).mean().values,
                    mode='lines',
                    marker_color='#E377C2',
                    name=f'{candle}_{d}MA',
                ),
                row=row,
                col=1,
                secondary_y=True
            )

            colors = [
                '#d3efd2' if o >= c else '#efd2d8' for o, c in zip(df[open], df[close])
            ]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df[name],
                    marker_color=colors,
                    name=name
                ),
                row=row,
                col=1,
                secondary_y=False
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[args[0]],
                    mode='lines',
                    name=args[1],
                    marker_color=args[2],
                    **kwargs
                ),
                row=row,
                col=1
            )

        return fig

    def plot_backtest_result(self, TestResult: object, title="Backtest Report"):
        '''將回測結果畫成圖表'''

        statement = TestResult.Statement.copy()
        statement.OpenDate = statement.OpenDate.apply(self._replaceString)
        statement.CloseDate = statement.CloseDate.apply(self._replaceString)

        daily_info = self._daily_info_processor(TestResult)

        N = len(self.Titles)
        n_tables = 3
        spec1 = [[{"type": "table"}]]*n_tables  # 3張表
        spec2 = [[{'secondary_y': True}]]*2  # TSE & OTC K線圖
        spec3 = [[{"type": "scatter"}]]*(N-n_tables-2)  # 其餘折線走勢圖
        fig = make_subplots(
            rows=N,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=self.Titles,
            specs=spec1 + spec2 + spec3,
            row_heights=[0.1, 0.2, 0.15] + [0.45/(N-n_tables)]*(N-n_tables)
        )
        fig = self.subplot_add_table(
            fig, TestResult.Configuration, row=1, col=1)
        fig = self.subplot_add_table(fig, TestResult.Summary, row=2, col=1)
        fig = self.subplot_add_table(fig, statement, row=3, col=1, height=40)
        for i, args in enumerate(self.TraceSettings):
            row = i+1+n_tables
            if i <= 1 or i == 3:
                fig = self._add_trace(fig, daily_info, row, *args)
            elif i == 4:
                for j, args_ in enumerate(args):
                    fig = self._add_trace(
                        fig, daily_info, row+j, *args_, fill='tozeroy')
            else:
                j = 0 if i == 2 else 1
                for args_ in args:
                    fig = self._add_trace(fig, daily_info, row+j, *args_)

        # figure layouts
        fig.update_layout(height=2400, width=1700, title_text=title)
        for i in range(N):
            fig.update_xaxes(
                rangeslider={'visible': True if i == N-1 else False},
                rangeslider_thickness=0.02 if i == N-1 else None,
                row=i+1,
                col=1
            )
        fig.layout.yaxis2.showgrid = False
        setattr(self.Figures, 'BacktestResult', fig)

        return self.Figures

    def plot_Ins_Outs(self, df: pd.DataFrame, testResult: object):
        '''畫圖: 個股進出場位置'''

        cols = ['name', 'Time', 'Open', 'High', 'Low', 'Close']
        statement = testResult.Statement
        statement['start'] = statement.groupby(
            'stock').OpenDate.transform(lambda x: x.min() - timedelta(days=5))
        statement['end'] = statement.groupby(
            'stock').CloseDate.transform(lambda x: x.max() + timedelta(days=5))
        tb = df[cols][
            (df.Time >= statement.start.min()) & (df.Time <= statement.end.max())]

        # figures = FigureInOut
        stocks = statement.stock.unique()
        N = stocks.shape[0]
        for i, s in enumerate(stocks):
            df1 = statement[statement.stock == s].copy()

            start = df1.start.values[0]
            end = df1.end.values[0]
            df2 = tb[(tb.name == s) & (tb.Time >= start) & (tb.Time <= end)]

            ins = df2[df2.Time.isin(df1.OpenDate)]
            ins = ins.set_index('Time').Low.to_dict()
            df1['Ins'] = df1.OpenDate.map(ins)

            outs = df2[df2.Time.isin(df1.CloseDate)]
            outs = outs.set_index('Time').High.to_dict()
            df1['Outs'] = df1.CloseDate.map(outs)

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df2.Time,
                        open=df2['Open'],
                        high=df2['High'],
                        low=df2['Low'],
                        close=df2['Close'],
                        name=f'{s}進出場點位',
                        increasing=dict(line=dict(color='Crimson')),
                        decreasing=dict(line=dict(color='LimeGreen'))
                    ),

                    go.Scatter(
                        x=df1.OpenDate,
                        y=df1.Ins*0.96,
                        customdata=np.stack(
                            (df1.OpenReason, df1.Ins),
                            axis=-1
                        ),
                        hovertemplate='%{x} <br>%{customdata[0]} <br>%{customdata[1]} 進場',
                        name='進場',
                        mode='markers',
                        marker=dict(
                            symbol='star-triangle-up',
                            color='MediumBlue',
                            size=10,
                            line=dict(color='MediumPurple', width=1)
                        ),
                    ),

                    go.Scatter(
                        x=df1.CloseDate,
                        y=df1.Outs*1.04,
                        customdata=df1[
                            ['CloseReason', 'Outs', 'profit', 'returns']].values,
                        hovertemplate='%{x} <br>%{customdata[0]} <br>出場: %{customdata[1]} <br>獲利: %{customdata[2]} (%{customdata[3]}%)',
                        name='出場',
                        mode='markers',
                        marker=dict(
                            symbol='star-triangle-down',
                            color='#17BECF',
                            size=10,
                            line=dict(color='MediumPurple', width=1)
                        ),
                    )
                ],
                # 設定 XY 顯示格式
                layout=go.Layout(
                    xaxis=go.layout.XAxis(tickformat='%Y-%m-%d %H:%M'),
                    yaxis=go.layout.YAxis(tickformat='.2f')
                )
            )

            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[14, 8], pattern="hour"),
                ]
            )
            setattr(self.Figures, f'fig{s}', fig)
            progress_bar(N, i)

        return self.Figures

    def save_figure(self, fig: object, filename='回測圖表'):
        '''輸出回測圖表'''

        folder_path = f'{self.DATAPATH}/回測報告/{TODAY_STR}-{filename}'
        file_handler.create_folder(folder_path)
        fig.BacktestResult.write_html(f'{folder_path}/回測結果.html')

        figures = [f for f in fig.__dict__ if 'fig' in f]
        for f in figures:
            fig.__dict__[f].write_html(f'{folder_path}/{f}.html')

        files = file_handler.listdir(folder_path, pattern='.html')
        for file in files:
            try:
                f = open(f'{folder_path}/{file}', "r+")
                fstr = f.read()
                ffilter = 'utf-8'
                fposition = fstr.find(ffilter)
                f.seek(fposition, 0)
                f.write('  Big5  ')
                f.close()
            except:
                print(f"\nCan't convert {file} encoding to Big5")

