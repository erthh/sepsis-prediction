# Import required libraries
import pickle
import copy
import pathlib
import dash
import math
import xgboost
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# Load data

#points = pickle.load(open(DATA_PATH.joinpath("points.pkl"), "rb"))
#logistic_model = pickle.load(open(DATA_PATH.joinpath("logistic_model.pkl"), "rb"))

# load the scaler model
loaded_scaler = pickle.load(open(DATA_PATH.joinpath("scaler.sav"), "rb"))
# load the XGBoost model
loaded_model = pickle.load(open(DATA_PATH.joinpath("xgb_9802_rfe.sav"), 'rb'))

#My data and variables
patient_data = pd.read_csv(DATA_PATH.joinpath("patient_data.csv"),low_memory=False)
patient_data = patient_data.drop(['ICULOS','Glucose'],axis=1)

#Declare variables 
signals = ['HR', 'MAP', 'O2Sat', 'SBP', 'Resp', 'DBP', 'Temp', 'Glucose', 'id']
rfe_feat = ['Temp_diff3', 'DBP_std', 'Age', 'Temp_diff2', 'Temp_diff4', 'Resp_min',
            'Resp_mean', 'Temp_diff1', 'Temp_diff5', 'Resp_max', 'MAP_min',
            'O2Sat_min', 'SBP_min', 'DBP_min', 'MAP_max', 'DBP_max', 'HR_max',
            'Temp_max', 'Temp_min', 'SBP_max', 'HR_mean', 'O2Sat_max']

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Sepsis Prediction",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Symtoms Overview", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("About us", id="learn-more-button"),
                            href="https://google.com",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Input section"),
                        html.Div([
                            html.H3('Select Patient ID'),
                            dcc.Dropdown(
                            id='Dropdown_patient_input',
                            options=[
                                    {'label': 'Patient ID: ' +str(i) , 'value' : i} for i in patient_data.id.unique()
                                ],
                            value='number',
                            clearable=False
                            )
                        ]),
                        html.Div(id="Patient_id_text"),

                        html.Div([
                            html.H3('Select Start Hour'),
                            dcc.Dropdown(
                            id='Dropdown_hour_input',
                            options=[],
                            value='number',
                            clearable=False
                            )
                        ]),
                        html.Div(id="ICU_stay_length"),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [   
                        html.Div(
                            [
                                html.Div(
                                    [html.H6("0",id="hr_text"), html.Div("Heart Rate (BPM)")],
                                    id="wells6",
                                    className="mini_container three columns",
                                ),
                                html.Div(
                                    [html.H6("0",id="temp_text"), html.Div("Temperature (Celsius)")],
                                    id="gas6",
                                    className="mini_container three columns",
                                ),
                                html.Div(
                                    [html.H6("0",id="o2sat_text"), html.Div("O2Sat (%)")],
                                    id="oil6",
                                    className="mini_container three columns",
                                ),
                                html.Div(
                                    [html.H6("0",id="resp_text"), html.Div("Respiratory Rate (BPM)")],
                                    id="water6",
                                    className="mini_container three columns",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [html.H6("0",id="map_text"), html.Div("MAP (mmHg)")],
                                    id="wells2",
                                    className="mini_container three columns",
                                ),
                                html.Div(
                                    [html.H6("0",id="sbp_text"), html.Div("SBP (mmHg)")],
                                    id="gas2",
                                    className="mini_container three columns",
                                ),
                                html.Div(
                                    [html.H6("0",id="dbp_text"), html.Div("DBP (mmHg)")],
                                    id="oil2",
                                    className="mini_container three columns",
                                ),
                                html.Div(
                                    [html.H6("0",id="maxhour_text"), html.Div("Max hours in ICU (hours)")],
                                    id="water2",
                                    className="mini_container three columns",
                                ),
                            ],
                            id="info-container2",
                            className="row container-display",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [html.H6("Probability of non sepsis"),html.P("0",id="nonsepsis_prob")],
                                    id="nonsepsis_container",
                                    className="mini_container six columns",
                                ),
                                html.Div(
                                    [html.H6("Probability of sepsis"),html.P("0",id="sepsis_prob")],
                                    id="sepsis_container",
                                    className="mini_container six columns",
                                ),
                                html.Div(
                                    [html.H6("Label of sepsis"),html.P("0",id="sepsis_label")],
                                    id="sepsis_label_container",
                                    className="mini_container four columns",
                                )
                            ],
                            id="info-container3",
                            className="row container-display",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="HR_chart",figure={'layout':{'title':'Other statistics'}})],
                    className="pretty_container six columns",
                ),
                html.Div(
                    [dcc.Graph(id="Temp_chart",figure={'layout':{'title':'Other statistics'}})],
                    className="pretty_container six columns",
                ),
            ],
            className="row flex-display",  
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="Resp_chart",figure={'layout':{'title':'Other statistics'}})],
                    className="pretty_container six columns",
                ),
                html.Div(
                    [dcc.Graph(id="O2Sat_chart",figure={'layout':{'title':'Other statistics'}})],
                    className="pretty_container six columns",
                ),
            ],
            className="row flex-display",
        ),
        html.H1("Dataset"),
        html.Div([
                dash_table.DataTable(
                id='table_temp',
                columns=[{"name": i, "id": i} for i in patient_data.columns],
                data=patient_data.to_dict('records'),   
                )
            ],className="row flex-display"
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


# Helper functions

def mean_rolling_window(df):
    data_mean = df[['id', 'HR', 'Resp']].groupby('id').rolling(window=6).mean()
    data_mean = data_mean.drop(columns='id')
    data_mean.columns += '_mean'
    return data_mean 

def std_rolling_window(df):
    data_std = df[['id', 'DBP']].groupby('id').rolling(window=6).std()
    data_std = data_std.drop(columns='id')
    data_std.columns += '_std'
    return data_std

def min_rolling_window(df):
    data_min = df[['id', 'Resp', 'MAP', 'O2Sat', 'SBP', 'DBP', 'Temp']].groupby('id').rolling(window=6).min()
    data_min = data_min.drop(columns='id')
    data_min.columns += '_min'
    return data_min

def max_rolling_window(df):
    data_max = df[['id', 'Resp', 'MAP', 'DBP', 'HR', 'O2Sat', 'SBP', 'Temp']].groupby('id').rolling(window=6).max()
    data_max = data_max.drop(columns='id')
    data_max.columns += '_max'
    return data_max

def diff1(x):
    return x[-1] - x[-2]

def find_diff1(df):
    data_diff1 = df[['id', 'Temp']].groupby('id').rolling(window=6).apply(diff1, raw=True)
    data_diff1 = data_diff1.drop(columns='id')
    data_diff1.columns += '_diff1'
    return data_diff1

def diff2(x):
    return x[-2] - x[3]

def find_diff2(df):

    data_diff2 = df[['id', 'Temp']].groupby('id').rolling(window=6).apply(diff2, raw=True)
    data_diff2 = data_diff2.drop(columns='id')
    data_diff2.columns += '_diff2'
    return data_diff2

def diff3(x):
    return x[-3] - x[-4]

def find_diff3(df):
    data_diff3 = df[['id', 'Temp']].groupby('id').rolling(window=6).apply(diff3, raw=True)
    data_diff3 = data_diff3.drop(columns='id')
    data_diff3.columns += '_diff3'
    return data_diff3

def diff4(x):
    return x[-4] - x[-5]

def find_diff4(df):
    data_diff4 = df[['id', 'Temp']].groupby('id').rolling(window=6).apply(diff4, raw=True)
    data_diff4 = data_diff4.drop(columns='id')
    data_diff4.columns += '_diff4'
    return data_diff4

def diff5(x):
    return x[-5] - x[0]

def find_diff5(df):
  data_diff5 = df[['id', 'Temp']].groupby('id').rolling(window=6).apply(diff5, raw=True)
  data_diff5 = data_diff5.drop(columns='id')
  data_diff5.columns += '_diff5'
  return data_diff5

def extracted_df(dataframe):
    # extract patterns for 6 hours
    data_mean = mean_rolling_window(dataframe)
    data_std = std_rolling_window(dataframe)
    data_min = min_rolling_window(dataframe)
    data_max = max_rolling_window(dataframe)
    data_diff1 = find_diff1(dataframe)
    data_diff2 = find_diff2(dataframe)
    data_diff3 = find_diff3(dataframe) 
    data_diff4 = find_diff4(dataframe)
    data_diff5 = find_diff5(dataframe)

    # append the extracted patterns together
    df = pd.concat([data_mean.reset_index(drop=True), data_std.reset_index(drop=True), data_min.reset_index(drop=True), \
                  data_max.reset_index(drop=True), data_diff1.reset_index(drop=True), data_diff2.reset_index(drop=True), \
                  data_diff3.reset_index(drop=True), data_diff4.reset_index(drop=True), data_diff5.reset_index(drop=True), \
                  dataframe[['Age']].reset_index(drop=True)], axis=1)

    # drop nan rows
    df = df.dropna()

    # standardize
    df_clean = loaded_scaler.transform(df)
    df_clean = pd.DataFrame(df_clean)
    df_clean.columns = df.columns

    # re-order the columns the same as how the model was trained
    df_clean = df_clean[rfe_feat]

    return df_clean

def get_statistic(df): 
    Max_hour_in_ICU = df['hour'].iloc[len(df['hour'])-1]
    # Age = df['Age'].iloc[1]
    mean_hr = round(df['HR'].mean(),3)
    mean_MAP = round(df['MAP'].mean(),3)
    mean_O2Sat = round(df['O2Sat'].mean(),3)
    mean_DBP = round(df['DBP'].mean(),3) 
    mean_SBP = round(df['SBP'].mean(),3)
    mean_Temp = round(df['Temp'].mean(),3)
    mean_Resp = round(df['Resp'].mean(),3)
    return mean_hr,mean_Temp,mean_O2Sat,mean_Resp,mean_MAP,mean_SBP,mean_DBP,Max_hour_in_ICU

def get_Line_Chart(df,col):
    #df = dataframe
    #col = specific column
    data = [go.Scatter(x=df['hour'] , y=df[col], name="trace_name",opacity=0.5)]
    #layout = {"title":str(col)+" chart vs hours",'x':'xsssssssssssssssss'}

    layout = dict(
        title = dict(text=str(col)+" chart vs hours"),
        xaxis = dict(title='Hours'),
        yaxis = dict(title=str(col))
        )
    return_output={
        "data":data,
        "layout":layout
    }
    return return_output

# ###########################
# #### Callback Function ####
# ###########################

@app.callback(
    Output('Dropdown_hour_input', 'options'),
    [Input('Dropdown_patient_input', 'value')]
    )
def update_dropdown_text(value):
    if isinstance(value,int):
        dataset = patient_data[patient_data['id']==value]
        option = [{'label': i, 'value': i} for i in dataset.iloc[6:len(dataset)].hour]
    else:
        option = []
    return option


@app.callback(
    Output('ICU_stay_length', 'children'),
    [Input('Dropdown_hour_input', 'value')]
    )
def update_dropdown_text(value):
    if isinstance(value,int):
        start_hour = value
        end_hour = value+2
        icu_stay_length = 'Patient stay in ICU from Hours ' + str(start_hour) + ' to ' + str(end_hour) +'.'
    else:
        icu_stay_length = ''
    return icu_stay_length

@app.callback(
    [Output('nonsepsis_prob', 'children'),Output('sepsis_prob', 'children'),Output('sepsis_label','children')],
    [Input('Dropdown_hour_input', 'value'),Input('Dropdown_patient_input', 'value')]
    )
def update_probability(hour_value,patient_id):
    if isinstance(hour_value,int):
        if hour_value <= 5:
            sepsis_prob = 0
            nonsepsis_prob = 0
        else:    
            hour_range = hour_value
            df_prob = patient_data[patient_data['id']==patient_id] 
            extracted_features = extracted_df(df_prob)
            predictions = loaded_model.predict(extracted_features)
            predictions_proba = loaded_model.predict_proba(extracted_features)
            predictions_label = predictions[hour_range-6]
            sepsis_prob = predictions_proba[hour_range-6,1]
            nonsepsis_prob = predictions_proba[hour_range-6,0]
    else:
        sepsis_prob = 0
        nonsepsis_prob = 0
        predictions_label = 0
    return sepsis_prob,nonsepsis_prob,predictions_label

@app.callback(
    [Output('hr_text', 'children'),
    Output('temp_text', 'children'),
    Output('o2sat_text', 'children'),
    Output('resp_text', 'children'),
    Output('map_text', 'children'),
    Output('sbp_text', 'children'),
    Output('dbp_text', 'children'),
    Output('maxhour_text', 'children'),
    ],
    [Input('Dropdown_patient_input', 'value')]
    )
def update_statistic_text(value):
    if isinstance(value,int):
        result_list = get_statistic(patient_data[patient_data['id']==value])
    else:
        result_list=[0,0,0,0,0,0,0,0]
    return result_list[0],result_list[1],result_list[2],result_list[3],result_list[4],result_list[5],result_list[6],result_list[7]

@app.callback(
    Output('HR_chart', 'figure'),
    [Input('Dropdown_patient_input', 'value')]
    )
def update_HR_chart(value):
    if isinstance(value,int):
        chart_result = get_Line_Chart(patient_data[patient_data['id']==value],"HR")
    else:
        chart_result = {}
    return chart_result

@app.callback(
    Output('Temp_chart', 'figure'),
    [Input('Dropdown_patient_input', 'value')]
    )
def update_Temp_chart(value):
    if isinstance(value,int):
        chart_result = get_Line_Chart(patient_data[patient_data['id']==value],"Temp")
    else:
        chart_result = {}
    return chart_result

@app.callback(
    Output('Resp_chart', 'figure'),
    [Input('Dropdown_patient_input', 'value')]
    )
def update_Resp_chart(value):
    if isinstance(value,int):
        chart_result = get_Line_Chart(patient_data[patient_data['id']==value],"Resp")
    else:
        chart_result = {}
    return chart_result

@app.callback(
    Output('O2Sat_chart', 'figure'),
    [Input('Dropdown_patient_input', 'value')]
    )
def update_O2Sat_chart(value):
    if isinstance(value,int):
        chart_result = get_Line_Chart(patient_data[patient_data['id']==value],"O2Sat")
    else:
        chart_result = {}
    return chart_result

# Main
if __name__ == "__main__":
    app.run_server(debug=True)






     