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

    def add_candlestick(self, fig, df, row, col):
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
            secondary_y=True
        )

        # plot MA
        for c, d in [('#447a9c', 5), ('#E377C2', 10)]:
            ma = df.Close.rolling(d).mean().values
            fig.add_trace(
                go.Scatter(
                    x=df.Time,
                    y=ma,
                    mode='lines+text',
                    marker_color=c,
                    name=f'{d}MA',
                    text=[f'{d}MA' if i == d else '' for i, _ in enumerate(ma)],
                    textfont=dict(color=c),
                    textposition='bottom right',
                ),
                row=row,
                col=col,
                secondary_y=True
            )

        # plot volume
        colors = [
            '#d3efd2' if o >= c else '#efd2d8' for o, c in zip(df.Open, df.Close)
        ]
        fig.add_trace(
            go.Bar(
                x=df.Time,
                y=df.Volume,
                marker_color=colors,
                name='Volume',
            ),
            row=row,
            col=col,
            secondary_y=False
        )

        # update axes settings
        fig.update_xaxes(
            rangeslider=dict(visible=False),
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                # dict(bounds=[14, 8], pattern="hour"),
            ],
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title=name,
            secondary_y=True,
            showgrid=True,
            tickformat=".0f",
            row=row,
            col=col
        )
        fig.update_yaxes(
            title="Volume",
            secondary_y=False,
            showgrid=False,
            row=row,
            col=col
        )
        return fig

    def add_line(self, fig: make_subplots, df: pd.DataFrame, row: int, col: int, settings: dict, **kwargs):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[settings['y']],
                mode='lines',
                name=settings['name'],
                marker_color=settings['marker_color'],
                **kwargs
            ),
            row=row,
            col=col
        )
        return fig
