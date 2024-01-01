import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def convert_encodings(filename):
    try:
        f = open(filename, "r+")
        fstr = f.read()
        ffilter = 'utf-8'
        fposition = fstr.find(ffilter)
        f.seek(fposition, 0)
        f.write('  Big5  ')
        f.close()
    except:
        print("\nCan't convert encoding to Big5")


def export_figure(fig, filename):
    if '.html' in fig:
        fig.write_html(filename)
    fig.write_image(filename, scale=5)


class SuplotHandler:
    def add_table(self, fig: make_subplots, df: pd.DataFrame, row: int, col: int, **cellargs):
        '''Subplot add table'''
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(df.columns),
                    font=dict(size=15),
                    align="center"
                ),
                cells=dict(
                    values=[df[k].tolist() for k in df.columns],
                    align="center",
                    font=dict(size=13),
                    **cellargs
                )
            ),
            row=row,
            col=col
        )
        return fig

    def add_candlestick(self, fig, df, row, col, plot_ma=True, plot_marker=False, plot_volume=False, **kwargs):
        if not df.shape[0]:
            return fig

        title = {'1': 'TWSE', '101': 'OTC'}
        name = df.name.unique()[0]
        name = title[name] if name in title else name

        # plot candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.Time,
                open=df.Open,
                high=df.High,
                low=df.Low,
                close=df.Close,
                name=name,
                increasing=dict(line=dict(color='#e63746')),
                decreasing=dict(line=dict(color='#42dd31')),
            ),
            row=row,
            col=col,
            # secondary_y=True
        )

        # plot MA
        if plot_ma:
            for c, d in [('#447a9c', 5), ('#E377C2', 10)]:
                ma = f'{d}MA'
                df[ma] = df.Close.rolling(d).mean()
                fig = self.add_line(
                    fig, df, row, col,
                    settings={'y': ma, 'name': ma, 'marker_color': c},
                    mode='lines+text',
                    text=[ma if i == d else '' for i in range(df.shape[0])],
                    textfont=dict(color=c),
                    textposition='bottom right',
                    secondary_y=True
                )

        if plot_marker:
            # Mark Buy/Sell points
            fig = self.add_marker(
                fig, df,
                row=row,
                col=col,
                settings=dict(color='#ff9f1a', symbol='triangle-up'),
                name=kwargs.get('marker_name1', 'Buy')
            )
            fig = self.add_marker(
                fig, df,
                row=row,
                col=col,
                settings=dict(color='#24799e', symbol='triangle-down'),
                name=kwargs.get('marker_name2', 'Sell')
            )

        # plot volume
        if plot_volume:
            colors = [
                '#d3efd2' if o >= c else '#efd2d8' for o, c in zip(df.Open, df.Close)
            ]
            fig.add_trace(
                go.Bar(
                    x=df.Time, y=df.Volume, marker_color=colors, name='Volume'),
                row=row,
                col=col,
                secondary_y=False
            )
            fig.update_yaxes(
                title="Volume",
                secondary_y=False,
                showgrid=False,
                row=row,
                col=col
            )
        return fig

    @staticmethod
    def add_line(fig: make_subplots, df: pd.DataFrame, row: int, col: int, settings: dict, **kwargs):
        mode = kwargs.get('mode', 'lines')
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[settings['y']],
                mode=mode,
                name=settings['name'],
                marker_color=settings['marker_color'],
                **kwargs
            ),
            row=row,
            col=col
        )
        return fig

    @staticmethod
    def add_marker(fig: make_subplots, df: pd.DataFrame, row: int, col: int, settings: dict, **kwargs):
        name = kwargs.get('name')
        fig.add_trace(
            go.Scatter(
                x=df.Time,
                y=df[name],
                mode='markers',
                name=name,
                marker=settings
            ),
            row=row,
            col=col
        )
        return fig
