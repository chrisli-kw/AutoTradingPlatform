import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from collections import namedtuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import PATH, TODAY, TODAY_STR, SelectMethods, StrategyNameList
from ..utils.time import TimeTool
from ..utils.file import FileHandler
from ..utils.orders import OrderTool
from ..utils.database import db, KBarTables
from ..utils.database.tables import SelectedStocks
from ..performance.base import convert_statement
from ..performance.backtest import BacktestPerformance
try:
    from ..scripts import __BacktestScripts__
except:
    logging.warning('Cannot import __BacktestScripts__ from package.')
    __BacktestScripts__ = None

bts = __BacktestScripts__.__dict__ if __BacktestScripts__ else {}


class PerformanceReporter(OrderTool, TimeTool, FileHandler):
    def __init__(self, account: str):
        self.account = account
        self.TablesFile = f'{PATH}/daily_info/{TODAY_STR[:-3]}-performance-{account}.xlsx'
        self.Tables = namedtuple(
            typename='Tables',
            field_names=['Configuration', 'Summary', 'Statement', 'Selection'],
            defaults=[None]*4
        )
        self.Scripts = {
            k[:-3]: v for k, v in bts.items()
            if ('T' in k or 'D' in k) and (k[:-3] in SelectMethods)
        }
        self.strategies = []

    def getStrategyList(self, df: pd.DataFrame):
        '''Get strategy list in code order'''
        
        strategies = pd.DataFrame([StrategyNameList.Code]).T.reset_index()
        strategies.columns = ['name', 'code']
        strategies = strategies[strategies.name.isin(df.Strategy)].name.values
        return strategies

    def getTables(self, config):

        def concat_strategy_table(results: dict, table_name: str):
            df = pd.DataFrame()
            for k, v in results.items():
                if table_name == 'Configuration':
                    temp = v.Configuration
                else:
                    temp = v.Summary
                temp = temp.rename(columns={'Description': k}).set_index('Content')
                df = pd.concat([df, temp], axis=1)
            return df.reset_index()

        init_position = int(config['INIT_POSITION'])

        df = self.read_statement(f'simulate-{self.account}')
        df = convert_statement(df, init_position=init_position)
        df = df[df.CloseTime.dt.month == TODAY.month]

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
            writer, encoding='utf-8-sig', index=False, sheet_name='Configuration')
        Tables.Summary.to_excel(
            writer, encoding='utf-8-sig', index=False, sheet_name='Summary')
        Tables.Statement.to_excel(
            writer, encoding='utf-8-sig', index=False, sheet_name='Statement')

        writer.save()
    
    def getSelections(self, statement):
        start = self.last_business_day(statement.OpenTime.values[0])
        if db.HAS_DB:
            df = db.query(
                SelectedStocks,
                SelectedStocks.date >= start
            )
        else:
            dir_path = f'{PATH}/selections/history'
            df = self.read_tables_in_folder(dir_path)
        df = df[
            df.Strategy.isin(statement.Strategy) &
            (df.date >= start)
        ]
        df['isLong'] = df.Strategy.map(
            statement.set_index('Strategy').isLong.to_dict()
        )
        return df
    
    def getKbarTable(self, df_select: pd.DataFrame):
        start = df_select.date.min()
        end = df_select.date.max() + timedelta(days=20)
        if db.HAS_DB:
            df = KBarTables['1D']
            df = db.query(
                df,
                df.date >= start,
                df.date <= end,
                df.name.in_(df_select.code)
            )
        else:
            dir_path = f'{PATH}/KBars/1D'
            df = self.read_tables_in_folder(dir_path)
            df = df[
                (df.date >= start) &
                (df.date <= end) &
                df.name.isin(df_select.code)
            ]
        df = df.sort_values(['name', 'date']).reset_index(drop=True)
        return df
    
    def plot_performance_report(self, Tables: namedtuple = None, save=True):
        if Tables is None:            
            df = pd.read_excel(self.TablesFile, sheet_name='Statement')

            df_summary = pd.read_excel(self.TablesFile, sheet_name='Summary')
            df_summary.iloc[22, 1:] = df_summary.iloc[22, 1:].apply(
                lambda x: float(x.replace('%', '')))
        else:
            df = Tables.Statement
            df_summary = Tables.Summary
        
        df.OpenTime = df.OpenTime.dt.date
        df.CloseTime = df.CloseTime.dt.date

        df_select = self.getSelections(df)
        table = self.getKbarTable(df_select)
        tb = df_select.drop_duplicates(['code', 'date', 'isLong'])
        profits = np.array([0.0]*tb.shape[0])
        for i, (code, day, is_long) in enumerate(zip(tb.code, tb.date, tb.isLong)):
            temp = table[(table.name == code) & (table.date >= day)].head(5)
            v1 = temp.Open.values[0]
            v2 = temp.Close.values[-1]
            m = 1 if is_long else -1
            profits[i] = 100*m*(v2-v1)/v1

        tb['profit'] = profits.round(4)
        tb['profit'].sum()

        # Make Plots
        subplot_titles=(
            '每日選股數',
            '累積獲利',
            '選股後5天漲幅',
            '實際交易獲利',
            "交易量",
            "獲利因子/盈虧比",
        )
        fig = make_subplots(rows=3, cols=2, subplot_titles=subplot_titles)

        
        if self.strategies == []:
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

        for stra, color in zip(self.strategies, colors):
            name=StrategyNameList.Code[stra]

            # 每日選股數
            tb1 = df_select[df_select.Strategy == stra]
            tb1 = tb1.groupby(['date', 'Strategy']).code.count()
            tb1 = tb1.reset_index()
            max_point = tb1.code.max()
            fig.add_trace(
                go.Scatter(
                    x=tb1.date,
                    y=tb1.code,
                    name=name,
                    showlegend=False,
                    marker_color=color,
                    mode='lines+text',
                    fill='tonexty',
                    stackgroup='one',
                    text=[name if x == max_point else '' for x in tb1.code],
                    textfont=dict(color='rgb(157, 42, 44)'),
                    textposition = 'top center',
                ),
                row=1,
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

            # 累積獲利
            tb2 = tb2.groupby('CloseTime').profit.sum().reset_index()
            tb2['cumsum_profit'] = tb2.profit.cumsum()
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
                    textposition = 'top left',
                ),
                row=1,
                col=2
            )
            fig.update_yaxes(tickvals=[0], row=1, col=2)

        # 交易量 & 獲利因子/盈虧比
        colors = [
            'rgb(252, 193, 74)',
            'hsl(51, 55%, 82%)', 
            'rgb(245, 126, 0)', 
            'rgb(17, 76, 95)', 
            'rgb(16, 154, 246)'
        ]
        for c, i in zip(colors, [19, 20, 21, 23, 24]):
            tb3 = df_summary.iloc[i, :]
            fig.add_trace(
                go.Bar(
                    x=[StrategyNameList.Code[s] for s in tb3[1:].index],
                    y=tb3[1:],
                    showlegend=False,
                    marker_color=c,
                ),
                row=3,
                col=1 if i < 22 else 2
            )

        start = df.OpenTime.min()
        end = df.CloseTime.max()
        title = f'{start} ~ {end} Trading Performance'
        fig.update_layout(
            title=title, 
            title_x=0.5,
            title_font=dict(size=23),
            bargap=0.15, 
            height=1200, 
            width=1000
        )
        fig.update_yaxes(title='Profit', tickvals=[0], row=1, col=2)
        fig.update_xaxes(title='Return(%)', row=2, col=1)
        fig.update_yaxes(title='Count', row=2, col=1)
        fig.update_xaxes(title='Return(%)', row=2, col=2)
        fig.update_yaxes(title='Profit', tickvals=[0], row=2, col=2)

        if save:
            fig.write_html(self.TablesFile.replace('xlsx', 'html'))
        return fig

