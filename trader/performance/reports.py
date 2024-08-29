import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from collections import namedtuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import tdp
from ..config import PATH, TODAY, TODAY_STR, SelectMethods, StrategyNameList
from ..utils import progress_bar
from ..utils.time import time_tool
from ..utils.file import file_handler
from ..utils.orders import OrderTool
from ..utils.database import db, KBarTables
from ..utils.database.tables import SelectedStocks
from .base import convert_statement
from .backtest import BacktestPerformance
from .charts import export_figure, convert_encodings, SuplotHandler
try:
    from ..scripts import backtest_config
except:
    logging.warning('Cannot import test scripts from package.')
    backtest_config = None


class FiguresSet:
    pass


class PerformanceReport(SuplotHandler, OrderTool):
    def __init__(self, account: str, market: str):
        self.account = account
        self.market = market
        self.set_report_scripts(backtest_config)
        self.TablesFile = f'{PATH}/daily_info/{TODAY_STR[:-3]}-{market}-performance-{account}.xlsx'
        self.Tables = namedtuple(
            typename='Tables',
            field_names=['Configuration', 'Summary', 'Statement', 'Selection'],
            defaults=[None]*4
        )
        self.strategies = []

    def set_report_scripts(self, report_scripts: object = None):
        if report_scripts:
            bts = report_scripts.__dict__
            if self.market == 'Stocks':
                self.Scripts = {
                    k[:-3]: v for k, v in bts.items()
                    if ('T' in k or 'D' in k) and (k[:-3] in SelectMethods)
                }
            else:
                self.Scripts = {
                    k: v for k, v in bts.items() if hasattr(v, 'market') and v.market == 'Futures'
                }
        else:
            self.Scripts = {}

    def getStrategyList(self, df: pd.DataFrame):
        '''Get strategy list in code order'''

        strategies = pd.DataFrame([StrategyNameList.Code]).T.reset_index()
        strategies.columns = ['name', 'code']
        strategies = strategies[strategies.name.isin(df.Strategy)].name.values
        return strategies

    def getTables(self, init_position, start=None, end=None):

        def concat_strategy_table(results: dict, table_name: str):
            df = pd.DataFrame()
            for k, v in results.items():
                if table_name == 'Configuration':
                    temp = v.Configuration
                else:
                    temp = v.Summary
                temp = temp.rename(
                    columns={'Description': k}).set_index('Content')
                df = pd.concat([df, temp], axis=1)
            return df.reset_index()

        df = self.read_statement(f'simulate-{self.account}')

        if not df.shape[0]:
            return None

        df = convert_statement(
            df,
            init_position=init_position,
            market=self.market,
            multipler={k: v.multipler for k, v in self.Scripts.items()}
        )

        # filter data
        if start is None and end is None:
            df = df[df.CloseTime.dt.month == TODAY.month]
        else:
            if not start:
                start = TODAY_STR[:-2] + '01'

            if not end:
                end = TODAY_STR

            df = df[(df.CloseTime >= str(start)) & (df.CloseTime <= str(end))]
        df['balance'] = init_position + df.profit.cumsum()
        df = df.reset_index(drop=True)

        self.strategies = self.getStrategyList(df)
        results = {}
        for stra in self.strategies:
            backtest_config = self.Scripts[stra]
            bp = BacktestPerformance(backtest_config)
            statement = df[df.Strategy == stra]

            result = dict(
                init_position=init_position,
                unit=int(init_position/100000),
                buyOrder='Close',
                statement=statement,
            )
            performance = bp.get_backtest_result(**result)
            results[stra] = performance

        df_config = concat_strategy_table(results, 'Configuration')
        df_summary = concat_strategy_table(results, 'Summary')
        return self.Tables(df_config, df_summary, df)

    def save_tables(self, Tables: namedtuple):
        writer = pd.ExcelWriter(self.TablesFile, engine='xlsxwriter')

        Tables.Configuration.to_excel(
            writer, index=False, sheet_name='Configuration')
        Tables.Summary.to_excel(writer, index=False, sheet_name='Summary')
        Tables.Statement.to_excel(writer, index=False, sheet_name='Statement')

        writer.close()

    def getSelections(self, statement):
        start = time_tool.last_business_day(statement.OpenTime.values[0])
        if db.HAS_DB:
            df = db.query(
                SelectedStocks,
                SelectedStocks.Time >= start
            )
        else:
            dir_path = f'{PATH}/selections/history'
            df = file_handler.read_tables_in_folder(dir_path)
        df = df[
            df.Strategy.isin(statement.Strategy) &
            (df.Time >= start)
        ]
        df['isLong'] = df.Strategy.map(
            statement.set_index('Strategy').isLong.to_dict()
        )
        return df

    def getKbarTable(self, df_select: pd.DataFrame):
        start = df_select.Time.min()
        end = df_select.Time.max() + timedelta(days=20)
        names = df_select.code.to_list() + ['1', '101']
        if db.HAS_DB:
            df = KBarTables['1D']
            df = db.query(
                df,
                df.Time >= start,
                df.Time <= end,
                df.name.in_(names)
            )
        else:
            dir_path = f'{PATH}/KBars/1D'
            df = file_handler.read_tables_in_folder(dir_path)
            df = df[
                (df.Time >= start) &
                (df.Time <= end) &
                df.name.isin(names)
            ]
        df = df.sort_values(['name', 'Time']).reset_index(drop=True)
        return df

    def plot_performance_report(self, Tables: namedtuple = None, save=True):
        if Tables is None:
            df = pd.read_excel(self.TablesFile, sheet_name='Statement')
            df_config = pd.read_excel(
                self.TablesFile, sheet_name='Configuration')
            df_summary = pd.read_excel(self.TablesFile, sheet_name='Summary')
            df_summary.iloc[22, 1:] = df_summary.iloc[22, 1:].apply(
                lambda x: float(x.replace('%', '')))
        else:
            df = Tables.Statement
            df_config = Tables.Configuration
            df_summary = Tables.Summary

        if not df.shape[0]:
            return None

        df.OpenTime = df.OpenTime.dt.date
        df.CloseTime = df.CloseTime.dt.date

        start = df.OpenTime.min()
        end = df.CloseTime.max()

        # Make Plots
        subplot_titles = (
            "TWSE",
            "OTC",
            'Number of Daily Selections',
            'Accumulated Profits',
            '5-day Percentage Change After Selection',
            'Actual Trading Profits',
            "Trading Volume",
            "Profit Factor/Profit Ratio",
        )
        specs = [[{'secondary_y': True}]*2] + [[{}, {}]]*3
        fig = make_subplots(
            rows=4, cols=2, subplot_titles=subplot_titles, specs=specs)

        if len(self.strategies) == []:
            self.strategies = self.getStrategyList(df)

        colors = [
            'rgb(22, 65, 192)',
            'rgb(16, 154, 246)',
            'rgb(49, 220, 246)',
            'rgb(49, 92, 246)',
            'rgb(49, 122, 246)',
            'rgb(28, 157, 212)',
            'rgb(89, 213, 238)',
            'rgb(49, 246, 240)',
            'rgb(128, 229, 249)',
            'rgb(139, 186, 234)',
        ][:len(self.strategies)]

        # Candlesticks
        if self.market == 'Stocks':
            df_select = self.getSelections(df)
            table = self.getKbarTable(df_select)
            tb = df_select.drop_duplicates(['code', 'Time', 'isLong'])
            profits = np.array([0.0]*tb.shape[0])
            for i, (code, day, is_long) in enumerate(zip(tb.code, tb.Time, tb.isLong)):
                temp = table[(table.name == code) & (
                    table.Time >= day)].head(5)
                v1 = temp.Open.values[0]
                v2 = temp.Close.values[-1]
                m = 1 if is_long else -1
                profits[i] = 100*m*(v2-v1)/v1

            tb['profit'] = profits.round(4)
            tb['profit'].sum()

            for name, col in [['1', 1], ['101', 2]]:
                temp = table[table.name == name].copy()
                fig = self.add_candlestick(fig, temp, 1, col, plot_volume=True)
                fig.update_xaxes(
                    rangeslider=dict(visible=False),
                    rangebreaks=[dict(bounds=["sat", "mon"])],
                    row=1,
                    col=col,
                )
        else:
            # TODO: 期貨多策略的K線圖
            if len(self.strategies) == 1:
                temp = tdp.convert_period_tick(start, end, scale='1D')
                fig = self.add_candlestick(fig, temp, 1, 1, plot_volume=True)
                fig.update_xaxes(
                    rangeslider=dict(visible=False),
                    rangebreaks=[dict(bounds=["sat", "mon"])],
                    row=1,
                    col=1,
                )
            else:
                logging.warning(
                    f'Cannot plot candlesticks chart if {len(self.strategies)} strategies')

        for stra, color in zip(self.strategies, colors):
            name = StrategyNameList.Code[stra]

            # 累積獲利
            init_position = int(df_config.loc[0, stra].replace(',', ''))
            tb2 = df[df.Strategy == stra].copy()
            tb2 = tb2.groupby('CloseTime').profit.sum().reset_index()
            tb2['cumsum_profit'] = 100*(tb2.profit.cumsum()/init_position)
            max_point = tb2.cumsum_profit.max()
            fig.add_trace(
                go.Scatter(
                    x=tb2.CloseTime,
                    y=tb2.cumsum_profit,
                    name=name,
                    showlegend=False,
                    marker_color=color,
                    mode='lines+text',
                    fill='tozeroy',
                    text=[name if x == max_point else '' for x in tb2.cumsum_profit],
                    textfont=dict(color='rgb(157, 42, 44)'),
                    textposition='top left',
                ),
                row=2,
                col=1
            )

            # 實際交易獲利(%)
            tb2 = df[df.Strategy == stra].copy()
            fig.add_trace(
                go.Scatter(
                    x=tb2.returns,
                    y=tb2.profit,
                    name=name,
                    showlegend=False,
                    marker_color=color,
                    mode='markers',
                ),
                row=2,
                col=2
            )

            if self.market == 'Futures':
                continue

            # 每日選股數
            tb1 = df_select[df_select.Strategy == stra]
            tb1 = tb1.groupby(['Time', 'Strategy']).code.count()
            tb1 = tb1.reset_index()
            max_point = tb1.code.max()
            fig.add_trace(
                go.Scatter(
                    x=tb1.Time,
                    y=tb1.code,
                    name=name,
                    showlegend=False,
                    marker_color=color,
                    mode='lines+text',
                    fill='tonexty',
                    stackgroup='one',
                    text=[name if x == max_point else '' for x in tb1.code],
                    textfont=dict(color='rgb(157, 42, 44)'),
                    textposition='top center',
                ),
                row=3,
                col=1
            )

            # 選股後5天漲幅(%)
            fig.add_trace(
                go.Histogram(
                    x=tb[tb.Strategy == stra].profit,
                    name=name,
                    showlegend=False,
                    marker_color=color,
                    nbinsx=50,
                ),
                row=3,
                col=2
            )

        # 交易量 & 獲利因子/盈虧比
        colors = [
            'rgb(252, 193, 74)',
            'hsl(51, 55%, 82%)',
            'rgb(245, 126, 0)',
            'rgb(17, 76, 95)',
            'rgb(16, 154, 246)'
        ]
        for c, i in zip(colors, [17, 18, 19, 21, 22]):
            tb3 = df_summary.iloc[i, :]
            fig.add_trace(
                go.Bar(
                    x=[StrategyNameList.Code[s] for s in tb3[1:].index],
                    y=tb3[1:],
                    showlegend=False,
                    marker_color=c,
                ),
                row=4,
                col=1 if i < 22 else 2
            )

        title = f'{start} ~ {end} {self.market} Trading Performance'
        fig.update_layout(
            title=title,
            title_x=0.5,
            title_font=dict(size=23),
            bargap=0.15,
            height=1500,
            width=1000
        )
        fig.update_yaxes(title='Profit(%)', row=2, col=2)
        fig.update_xaxes(title='Return(%)', row=3, col=1)
        fig.update_yaxes(title='Count', row=3, col=1)
        fig.update_xaxes(title='Return(%)', row=3, col=2)
        fig.update_yaxes(title='Profit', tickvals=[0], row=3, col=2)

        if save:
            export_figure(fig, self.TablesFile.replace('xlsx', 'jpg'))
        return fig


class BacktestReport(SuplotHandler):
    def __init__(self, backtestScript) -> None:
        self.Figures = FiguresSet
        self.Script = backtestScript
        self.DATAPATH = f'{PATH}/backtest'
        self.col_maps = {
            'chance': 'first',
            'n_stock_limit': 'first',
            'n_stocks': 'last',
            'nClose': 'sum',
            'balance': 'last',
            'nOpen': 'sum',
            'profit': 'sum',
        }
        self.Titles = (
            # tables
            'Backtest Settings', '',
            'Summary', '',
            'Transaction Detail', '',

            # candles
            'TWSE', 'OTC',

            # lines/scatters
            'Put/Call Ratio', 'Changes in Opens & Closes',
            'Balance', 'Accumulated Profit/Loss',
            'Changes in Portfolio Control', ''
        )

    def _replaceString(self, x: str):
        return str(x).replace(' ', '<br>').replace('00.000000000', '00')

    def _daily_info_processor(self, TestResult: object):
        profit = TestResult.Statement.groupby('CloseTime').profit.sum()
        profit = profit.to_dict()

        df = TestResult.DailyInfo
        df['profit'] = df.index.map(profit).fillna(0).values
        df = df.resample('1D', closed='left',
                         label='left').apply(self.col_maps)
        df = df.dropna()
        df['profits'] = (df.profit*(df.profit > 0)).cumsum()
        df['losses'] = (df.profit*(df.profit <= 0)).cumsum()
        return df

    def plot_backtest_result(self, TestResult: object, KBars: dict, title="Backtest Report"):
        '''將回測結果畫成圖表'''

        if TestResult.Statement is None:
            return self.Figures

        statement = TestResult.Statement.copy()
        statement.OpenTime = statement.OpenTime.apply(self._replaceString)
        statement.CloseTime = statement.CloseTime.apply(self._replaceString)

        daily_info = self._daily_info_processor(TestResult)

        N = 7
        n_tables = 3
        spec1 = [[{"type": "table", "colspan": 2}, {}]]*n_tables  # 3張表
        spec2 = [[{'secondary_y': True}]*2]  # TSE & OTC K線圖
        spec3 = [[{"type": "scatter"}]*2]*(N-n_tables-1)  # 其餘折線走勢圖
        fig = make_subplots(
            rows=N,
            cols=2,
            vertical_spacing=0.02,
            subplot_titles=list(self.Titles),
            specs=spec1 + spec2 + spec3,
            row_heights=[0.1, 0.15, 0.15] + [0.6/(N-n_tables)]*(N-n_tables)
        )
        fig = self.add_table(fig, TestResult.Configuration, row=1, col=1)
        fig = self.add_table(fig, TestResult.Summary, row=2, col=1)
        fig = self.add_table(fig, statement, row=3, col=1, height=40)

        # TSE/OTC candlesticks
        if self.Script.market == 'Stocks':
            for col, (a, b) in enumerate([('1', 'TWSE'), ('101', 'OTC')]):
                temp = KBars['1D'].copy()
                temp = temp[temp.name == a].sort_values('Time')
                temp['name'] = b
                fig = self.add_candlestick(
                    fig, temp, row=4, col=col+1, plot_volume=True)
                fig.update_xaxes(
                    rangeslider=dict(visible=False),
                    rangebreaks=[dict(bounds=["sat", "mon"])],
                    row=4,
                    col=col+1,
                )
                fig.update_yaxes(
                    title=b,
                    secondary_y=True,
                    showgrid=True,
                    tickformat=".0f",
                    row=4,
                    col=col+1
                )
        else:
            fig = self.add_candlestick(
                fig, KBars['1D'].copy(), row=4, col=1, plot_volume=True)

        # Put/Call Ratio
        if 'put_call_ratio' in KBars:
            df_pcr = pd.DataFrame(KBars['put_call_ratio']).T
            df_pcr['pc115'] = 115
            for args in [
                dict(y='PutCallRatio', name='Put/Call Ratio',
                     marker_color='#ff9f1a'),
                dict(y='pc115', name='多空分界', marker_color='#68c035'),
            ]:
                fig = self.add_line(
                    fig,
                    df_pcr,
                    row=5,
                    col=1,
                    settings=args
                )

        # Changes in Opens & Closes
        for args in [
            dict(y='nOpen', name='Opens', marker_color='#48b2ef'),
            dict(y='nClose', name='Closes', marker_color='#7F7F7F'),
            dict(y='n_stocks', name='in-stock', marker_color='#ef488e')
        ]:
            fig = self.add_line(fig, daily_info, row=5, col=2, settings=args)

        # Accumulated Balance
        args = dict(y='balance', name='Accumulated Balance',
                    marker_color='#d3503c')
        fig = self.add_line(fig, daily_info, row=6, col=1, settings=args)

        # Accumulated Profit/Loss
        for args in [
            dict(y='profits', name="Accumulated Profit", marker_color='#c25656'),
            dict(y='losses', name="Accumulated Loss", marker_color='#9ad37e')
        ]:
            fig = self.add_line(
                fig, daily_info, row=6, col=2, settings=args, fill='tozeroy')

        # Changes in Portfolio Control
        for args in [
            dict(y='chance', name='Opens Available', marker_color='#48efec'),
            dict(y='n_stock_limit', name='Opens Limit', marker_color='#b1487f')
        ]:
            fig = self.add_line(fig, daily_info, row=7, col=1, settings=args)

        # figure layouts
        fig.update_layout(height=2400, width=1700, title_text=title)
        setattr(self.Figures, 'BacktestResult', fig)

        return self.Figures

    def plot_Ins_Outs(self, df: pd.DataFrame, testResult: object):
        '''畫圖: 個股進出場位置'''

        cols = ['name', 'Time', 'Open', 'High', 'Low', 'Close']
        statement = testResult.Statement
        statement['start'] = statement.groupby(
            'stock').OpenTime.transform(lambda x: x.min() - timedelta(days=5))
        statement['end'] = statement.groupby(
            'stock').CloseTime.transform(lambda x: x.max() + timedelta(days=5))
        tb = df[cols][
            (df.Time >= statement.start.min()) & (df.Time <= statement.end.max())]

        stocks = statement.stock.unique()
        N = stocks.shape[0]
        for i, s in enumerate(stocks):
            df1 = statement[statement.stock == s].copy()

            start = df1.start.values[0]
            end = df1.end.values[0]
            df2 = tb[(tb.name == s) & (tb.Time >= start) & (tb.Time <= end)]

            ins = df2[df2.Time.isin(df1.OpenTime)]
            ins = ins.set_index('Time').Low.to_dict()
            df1['Ins'] = df1.OpenTime.map(ins)

            outs = df2[df2.Time.isin(df1.CloseTime)]
            outs = outs.set_index('Time').High.to_dict()
            df1['Outs'] = df1.CloseTime.map(outs)

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
                        x=df1.OpenTime,
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
                        x=df1.CloseTime,
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
        file_handler.Operate.create_folder(folder_path)
        export_figure(fig.BacktestResult, f'{folder_path}/回測結果.html')

        figures = [f for f in fig.__dict__ if 'fig' in f]
        for f in figures:
            export_figure(fig.__dict__[f], f'{folder_path}/{f}.html')

        files = file_handler.Operate.listdir(folder_path, pattern='.html')
        for file in files:
            convert_encodings(f'{folder_path}/{file}')
