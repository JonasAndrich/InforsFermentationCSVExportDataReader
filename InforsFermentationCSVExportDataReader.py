# -----------------------------------------------------------
# Visualization of fermentation data in a Dash-Web-App
# Reads the CSV-Files specificly exported by the Infors eve-Software
# CSV-Files exported with ";" as separation symbol, "," as decimal symbol and "ISO-8859-1"-encoding.
# (C) 2021 Jonas Andrich, Hamburg, Germany
# Released under GNU Public License (GPL)
# email Jonas.Andrich@gmail.com
# -----------------------------------------------------------

# Dash-Compounds
# from jupyter_dash import JupyterDash
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
from dash_table.Format import Format
# from dash.exceptions import PreventUpdate

# Pandas
import pandas as pd

# numpy
import numpy as np

# Plotly
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# System
import base64
# import datetime
import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

server = app.server

app.layout = html.Div([
    html.H1("Infors Fermentation CSV Data export Reader"),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),

    html.Div(html.Div([dcc.Dropdown(id='Y-Value1'), dcc.Dropdown(id='Y-Value2'), dcc.RadioItems(id='X-Value')],
                      id='figure-input')),

    dcc.Store(id='local', storage_type='session'),
])


@app.callback(Output('figure-input', 'children'),
              Output('local', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              prevent_initial_call=True)
def update_output(contents, name, date):
    print(name)
    if contents is not None:
        df = parse_contents(contents)
        children = html.Div([
            html.H2(name),
            html.Label([('Primary Y-Axis'),
                        dcc.Dropdown(
                            id='Y-Value1',
                            options=[
                                {'label': c, 'value': c}
                                for c in (df.columns.values)
                            ],
                            value=['pO2, %',
                                   # 'Glycerol, g/l',
                                   # "absSubstrate_added_until_now, g"
                                   # 'pH, -',
                                   ],
                            multi=True
                        ),

                        ]),

            html.Label([('Select Secondary Y-Axis'),
                        dcc.Dropdown(
                            id='Y-Value2',
                            options=[
                                {'label': c, 'value': c}
                                for c in (df.columns.values)
                            ],

                            value=['Air Flow, l/min',
                                   # 'µ',
                                   # 'yield X/S',
                                   'GenericGasAnalyser.Exit CO2, %'
                                   ],
                            multi=True
                        ),
                        ]),

            html.Label([('Select X'),
                        dcc.RadioItems(
                            id='X-Value',
                            options=[

                                {'label': 'Date Local Time', 'value': 'Date Local Time'},
                                {'label': 'Batch Time, h', 'value': 'Batch Time, h'},
                            ],
                            value='Batch Time, h'

                        ),
                        ]),

            html.Div(id='FermentationDataGraph'),
            html.Div(id='table'),

        ])

        return children, df.to_json(date_format='iso', orient='split')


# callback to update graph
@app.callback(
    Output('FermentationDataGraph', 'children'),
    Input('Y-Value1', 'value'),
    Input('Y-Value2', 'value'),
    Input('X-Value', 'value'),
    Input('local', 'data'),
    prevent_initial_call=True
)
def update_figure(YValue1, YValue2, XValue, data):
    df = pd.read_json(data, orient='split')

    # Groups of Parameter ranges
    # ph 1-7
    # Exit CO2 0,2 - 6
    # L/min 0-2
    # RQ 0 - 5
    # Exit O2 12-20
    # Temperature 20- 37
    # O2 0-100
    # Waage 0-1000 g
    # Stirrer 600-1200

    # Set x-axis data and title based on selection

    if XValue == 'Date Local Time':
        X = df.index.tolist()
    else:
        X = df[XValue].tolist()
    # print(X)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(xaxis_title=XValue)

    # List for Titles of the Y-Axis elements
    primary_title_list = []
    secondary_title_list = []

    for element in YValue1:
        Y = df[element].tolist()
        fig.add_trace(go.Scatter(x=X, y=Y,
                                 mode='markers',
                                 name=element,
                                 marker=dict(size=5)
                                 ),
                      secondary_y=False)
        primary_title_list.append(element)

    for element in YValue2:
        Y = df[element].tolist()
        fig.add_trace(go.Scatter(x=X, y=Y,
                                 mode='markers',
                                 name=element,
                                 marker=dict(size=5)
                                 ),
                      secondary_y=True)
        secondary_title_list.append(element)

    # Merge and set primary y-axes titles
    fig.update_yaxes(
        title_text='<br> '.join(primary_title_list),
        secondary_y=False)

    # Merge and set secondary y-axes titles
    fig.update_yaxes(
        title_text='<br> '.join(secondary_title_list),
        secondary_y=True)

    # Formatieren der Graphen-Legende
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    # Größe des Graphen und Zeit Dauer Veränderung wenn Werte geändert werden
    fig.update_layout(
        transition_duration=500,
        autosize=True,
        # width=500,
        # height=700,
    )

    children = dcc.Graph(
        figure=fig
    )
    return children


# Adding the Table
@app.callback(
    Output('table', 'children'),
    Input('local', 'data'),
    prevent_initial_call=True
)
def add_table(data):
    df = pd.read_json(data, orient='split')
    if 'BTM, g/l' in list(df.columns.values):
        children = dash_table.DataTable(id='table',
                                        columns=[{"name": i, "id": i, "type": 'numeric', "format": Format(precision=2)}
                                                 for i in df.columns],
                                        data=df[df['BTM, g/l'].notnull()].to_dict('records'), )

        return children


def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        data = pd.read_csv(
            io.StringIO(decoded.decode(encoding='utf-8', errors='ignore')),
            sep=";",
            decimal=",",
            header=1,
            encoding='utf-8')

        # do some operations on data
        data['Batch Time, h'] = data['Batch Time, sec'] / 3600

        # conversion to of column values to proper datetime objects
        data['Date Local Time'] = pd.to_datetime(data['Date Local Time'], dayfirst=True)

        # setting date time as index
        data = data.set_index('Date Local Time', drop=False)
        # Remove columns with no content
        data.dropna(how='all', axis=1, inplace=True)
        print(data)

        if 'BTM, g/l' in list(data.columns.values):
            # Infors does extrapolate Offline-Parameters in a linear manner instead of taking it as a single measuring
            # point in time. To get rid of this behaviour, all duplicates are dropped.
            data['BTM, g/l'] = data['BTM, g/l'].dropna(how='any').drop_duplicates(keep='first', inplace=False)

            # calculate my
            data = calculate_my(data)

        if 'Glycerol, g/l' in list(data.columns.values):
            # Problem mit zero values beheben
            data['Glycerol, g/l'] = data['Glycerol, g/l'].dropna(how='any').drop_duplicates(keep='first', inplace=False)
            data = calculate_yield(data)

        # print(list(data.columns.values))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return data


def calculate_my(df):
    # calculate time difference only between rows with BTM-Values
    # get timestamp-data to do calculations more easily
    df['tvalue'] = df.index

    df['BTMt0'] = df['BTM, g/l'].dropna(how='any').shift()
    df["BTM/BTMt0"] = df['BTM, g/l'] / df['BTMt0']
    df['natural_BTM/BTMt0'] = np.log(df["BTM/BTMt0"])

    # get the previous time stamp value to be able to calculate the time difference t1-t0
    df['BTM_t0'] = df[df['BTM, g/l'].notnull()]['tvalue'].shift()

    # calculate the actual time difference t1-t0
    df['BTM_t1-t0'] = df[df['BTM, g/l'].notnull()]['tvalue'] - df[df['BTM, g/l'].notnull()]['BTM_t0']

    # calculate an actual hour value from the timedelta
    # pd.Timedelta(hours=1) kann weg?
    df['BTM_t1-t0_h'] = df['BTM_t1-t0'] / np.timedelta64(1, 'h')

    # Calculate the µ
    df['µ'] = df['natural_BTM/BTMt0'] / df['BTM_t1-t0_h']
    df.drop(columns=['BTMt0',
                     'BTM_t1-t0',
                     "BTM/BTMt0",
                     'natural_BTM/BTMt0',
                     'BTM_t0',
                     'BTM_t1-t0',
                     'BTM_t1-t0_h'
                     ],
            inplace=True)

    return df


def calculate_yield(df):
    # calculate time difference only between rows with BTM-Values
    # get timestamp-data to do calculations more easily
    df['tvalue'] = df.index
    # calculate the approximate instant cultivation volume
    # 'Weight, kg' is actually -g
    df['volumne'] = 0.4 + (-df['Weight, kg'] / 1110)

    # biomass
    df["absBTM, g"] = df['BTM, g/l'] * df['volumne']
    df['absBTMt0'] = df["absBTM, g"].dropna(how='any').shift()
    df["deltaabsBTM"] = df["absBTM, g"] - df['absBTMt0']

    # substrate

    # Determine the absolute added substrate amount
    df["absSubstrate_added_until_now, g"] = 0.400 * 20 + (-df['Weight, kg'] / 1110) * 400

    # get the previous Substrate to be able to calculate the difference
    df["absSubstrate_added_until_prev"] = df[df['BTM, g/l'].notnull()]["absSubstrate_added_until_now, g"].shift()

    # absolute amount of glycerol in the System present
    df["absSubstrate_now_present, g"] = df['Glycerol, g/l'].dropna(how='any') * df['volumne']
    df["absSubstrate_prev_present, g"] = df[df['BTM, g/l'].notnull()]["absSubstrate_now_present, g"].shift()

    df["absSubstrate_comsumed_until_now, g"] = df["absSubstrate_added_until_now, g"] - df["absSubstrate_now_present, g"]

    # calculate the absolute consumed glycerol between timepoints
    df["deltaabsSubstratecomsumed, g"] = (df["absSubstrate_comsumed_until_now, g"]) - (
                df["absSubstrate_added_until_prev"] -
                df["absSubstrate_prev_present, g"])

    df['yield X/S'] = (df["deltaabsBTM"] / df["deltaabsSubstratecomsumed, g"])

    df['yield X/S cumulative'] = df["absBTM, g"] / (df["absSubstrate_comsumed_until_now, g"])

    # Berechnungen für die Yield Batch-Phase und Yield Feed-Phase
    # Für Yield in der Batchphase (Conc BTM/ Conc Glycerol)

    # Biomass end of Batchphase
    # BTMabsendofBatch = df[df["Phase"]=="Feedphase"]['BTM, g/l'].dropna(how='any').iloc[0]
    # print(BTMabsendofBatch )

    # SubstrateabsConsumedEndofBatch = df[df["Phase"]=="Feedphase"]["absSubstrate_comsumed_until_now, g"].dropna(how='any').iloc[0]
    # print (SubstrateabsConsumedEndofBatch)
    # YXSBatch = BTMabsendofBatch/SubstrateabsConsumedEndofBatch

    YXSBatch = df[df["Phase"] == "Feedphase"]['yield X/S cumulative'].dropna(how='any').iloc[0]

    # Für Yield in der Feed-Phase
    BTMabsBatch = (df[df["Phase"] == "Feedphase"]['BTM, g/l'].dropna(how='any').iloc[0] * 0.4)
    BTMabsfinal = df['volumne'].iloc[-1] * df[df["Phase"] == "Feedphase"]['BTM, g/l'].dropna(how='any').iloc[-1]

    # Calculate absolut Biomass created in Feed-Phase
    BTMabsFeed = BTMabsfinal - BTMabsBatch

    # Calculate absolute substrate Addition
    df['GlycerolFed, g'] = (-df['Weight, kg'] / 1110) * 400
    df['GlycerolFedComsumed, g'] = df['GlycerolFed, g'].dropna(how='any') - df["absSubstrate_now_present, g"]

    YXSFeed = BTMabsFeed / df['GlycerolFedComsumed, g'].dropna(how='any').iloc[-1]

    print(YXSBatch)
    print(YXSFeed)

    # Cleanup
    df.drop(columns=["absSubstrate_added_until_prev",
                     'absBTMt0',
                     "absSubstrate_prev_present, g",
                     "deltaabsSubstratecomsumed, g",
                     'absBTMt0',
                     "deltaabsBTM",
                     'tvalue'], inplace=True)

    return df


if __name__ == '__main__':
    app.run_server(debug=True)