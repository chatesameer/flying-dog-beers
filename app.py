import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pickle
import json
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
# model & metric imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

########### Define your variables ######
myheading1='TTO Customer Prediction'
image1='ames_welcome.jpeg'
tabtitle = 'Prediction'
githublink = 'https://github.com/chatesameer/flying-dog-beers'

#importing datset with completed_flag = 0 (non-customers)
url_0 = '../../../Intuit_Vertica/tto_0_df.csv'
tto_0_df = pd.read_csv(url_0, low_memory=False)
#importing datset with completed_flag = 1 (customers)
url_1 = '../../../Intuit_Vertica/tto_1_df.csv'
tto_1_df = pd.read_csv(url_1, low_memory=False)

tto_0_df_subset = tto_0_df[['auth_id','completed_flag', 'fiveYearTenureName', 'basic_segment_type_name', 'ageInYears', 'agiBandName', 'isRenter', 'priorYearIsRenter', 'priorYearAgiBandName', 'lostIncome', 'priorYearFilingStatusName', 'filingStatusName', 'priorYearStartDate', 'priorYearCompleteSku']].copy(deep=True)
tto_1_df_subset = tto_1_df[['auth_id','completed_flag', 'fiveYearTenureName', 'basic_segment_type_name', 'ageInYears', 'agiBandName', 'isRenter', 'priorYearIsRenter', 'priorYearAgiBandName', 'lostIncome', 'priorYearFilingStatusName', 'filingStatusName', 'priorYearStartDate', 'priorYearCompleteSku']].copy(deep=True)
tto_df_master = tto_0_df_subset.append(tto_1_df_subset)
# priorYearStartDate is a datetime column. Let's extract the month from the date.
tto_df_master['priorYearStartMonth'] = pd.DatetimeIndex(tto_df_master['priorYearStartDate']).month
# I want the month to be month name in order to avoid creating weighted differences by using month number
tto_df_master['priorYearStartMonth'] = pd.to_datetime(tto_df_master['priorYearStartMonth'], format='%m').dt.month_name().str.slice(stop=3)
# Since there are only 94k records with NaN values, I will drop these rows.
tto_df_master.dropna(subset=['priorYearStartMonth'], inplace=True)
tto_df_master.reset_index(drop=True)
# Now lets fill all NaN values with our observations
tto_df_master.fillna({'ageInYears': 42, 'agiBandName': '100K+', 'isRenter': 0, 'priorYearIsRenter': 0, 'priorYearAgiBandName': '100K+', 'priorYearFilingStatusName': 'Single', 'filingStatusName': 'Single', 'priorYearCompleteSku': 'NotCompleted'}, inplace=True)
# There are no null values remaining now
# Lets reset index to auth id and do some further analysis on few other columns
tto_df_master.set_index('auth_id', inplace=True)
# For basic_segment_type_name, we see that there are a lot of different values.
# This will create too many columns after one-hot encoding.
# Hence, lets put a threshold and make all values below the threshold as 'Others'
basicSeg_counts = tto_df_master['basic_segment_type_name'].value_counts()
basicSeg_lowCounts = basicSeg_counts[basicSeg_counts < 100000].index
tto_df_master['basic_segment_type_name'].replace(to_replace=list(basicSeg_lowCounts), value='Other', inplace=True)

#One hot encoding
onehot_tenure = pd.get_dummies(tto_df_master['fiveYearTenureName'], prefix='5yrTenure')
onehot_basicSeg = pd.get_dummies(tto_df_master['basic_segment_type_name'], prefix='basicSeg')
onehot_agiBandName = pd.get_dummies(tto_df_master['agiBandName'], prefix='agiBandName')
onehot_pyAgiBandName = pd.get_dummies(tto_df_master['priorYearAgiBandName'], prefix='pyAgiBandName')
onehot_pyFilingStatusName = pd.get_dummies(tto_df_master['priorYearFilingStatusName'], prefix='pyFilingStatusName')
onehot_filingStatusName = pd.get_dummies(tto_df_master['filingStatusName'], prefix='filingStatusName')
onehot_pyCompleteSku = pd.get_dummies(tto_df_master['priorYearCompleteSku'], prefix='pyCompleteSku')
onehot_pyStartMonth = pd.get_dummies(tto_df_master['priorYearStartMonth'], prefix='pyStartMonth')

# Merging all onehot encoded columns to the tto_df_master dataframe
tto_df_master = pd.concat([tto_df_master, onehot_tenure, onehot_basicSeg, onehot_pyFilingStatusName, onehot_filingStatusName, onehot_pyAgiBandName, onehot_agiBandName, onehot_pyCompleteSku, onehot_pyStartMonth], axis=1)

# Dropping all columns which were converted into one hot encoded columns and also the priorYearStartDate columns since we converted it into month
tto_df_master.drop(labels=['fiveYearTenureName', 'basic_segment_type_name', 'priorYearFilingStatusName', 'filingStatusName', 'priorYearAgiBandName', 'agiBandName', 'priorYearCompleteSku', 'priorYearStartMonth', 'priorYearStartDate'], axis=1, inplace=True)

# Preparing feature_set and applying Random Forest algorithm
feature_set = list(tto_df_master.columns)

#remove completed_flag element from the list of feature_set since that is our prediction
feature_set.remove('completed_flag')

#Assign the features and targets to appropriate variables
X = tto_df_master[feature_set]
y = tto_df_master['completed_flag']

# split the data into training and testing data (default is 75/25)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# instantiate the model
rfClassifier = RandomForestClassifier(n_estimators = 250, random_state = 1)

# fit with data
rfClassifier.fit(X_train, y_train)

# Write a function to preprocess and predict
# Remember: the order of arguments must match the order of features
features = ['fiveYearTenureName','basic_segment_type_name','ageInYears','agiBandName','priorYearAgiBandName','isRenter','priorYearIsRenter','lostIncome','filingStatusName','priorYearFilingStatusName','priorYearCompleteSku','priorYearStartMonth']

def tto_customer_prediction(listOfArgs):
    try:
        # the order of the arguments must match the order of the features
        input_df = pd.DataFrame(columns=features) 
        input_df.loc[0] = listOfArgs
        # Creating new columns in df similar to one-hot encoded columns and assign a value of 1
        input_df['basicSeg_'+input_df['basic_segment_type_name']]=1
        input_df['agiBandName_'+input_df['agiBandName']]=1
        input_df['pyAgiBandName_'+input_df['priorYearAgiBandName']]=1
        input_df['filingStatusName_'+input_df['filingStatusName']]=1
        input_df['pyFilingStatusName_'+input_df['priorYearFilingStatusName']]=1
        input_df['pyCompleteSku_'+input_df['priorYearCompleteSku']]=1
        input_df['pyStartMonth_'+input_df['priorYearStartMonth']]=1

        #drop columns that were converted earlier
        input_df.drop(labels=['fiveYearTenureName', 'basic_segment_type_name', 'priorYearFilingStatusName', 'filingStatusName', 'priorYearAgiBandName', 'agiBandName', 'priorYearCompleteSku', 'priorYearStartMonth'], axis=1, inplace=True)

        # Create a new dataframe that will go as input to Random Forest classifier
        X_df = pd.DataFrame(columns=feature_cols) 
        # Assign 0 to all columns in the first row of the new dataframe
        X_df.loc[len(X_df)]=0
        # For loop will run through each of the columns in df and if it has a match in X_df, it will assign the same value to the column in X_df
        for item in list(input_df.columns):
            for x in list(X_df.columns):
                if item == x:
                    X_df[x]=input_df[item]

        # Run Random Forest classifier prediction on X_df
        prediction=rfClassifier.predict(X_df)
        if prediction[0] == 0:
            return 'This profile will NOT be a TTO customer'
        else:
            return 'This profile will be a TTO customer'
    
    except:
        return 'Invalid inputs! Please try with different values'

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading1),

    html.Div([
        html.Div(
            [dcc.Graph(id='fig1',style={'width': '90vh', 'height': '90vh'}),
            ], className='eight columns'),
        html.Div([
                html.H3("Features"),
                html.Div('fiveYearTenureName'),
                dcc.Dropdown(id='fiveYearTenureName',
                    options=[{'label': i, 'value': i} for i in ['Vet','ANC Vet','ANC New','1 yr','WB 1 yr','WB 2 yr','Prospect 1 yr','WB 3 yr','WB 4 yr','Prospect 3 yr','Prospect 2 yr','WB 5 yr']],
                    value='ANC New'),
                html.Div('basic_segment_type_name'),
                dcc.Dropdown(id='basic_segment_type_name',
                    options=[{'label': i, 'value': i} for i in ['Deluxe ANC','Deluxe','Free ANC','Premier','SE ANC','Free','SE','Premier ANC','TTLD ANC','Partner','TTLD','FFA ANC','Other']],
                    value='Deluxe ANC'),
                html.Div('ageInYears'),
                dcc.Input(id='ageInYears', value=26, type='number', min=0, max=100, step=10),
                html.Div('agiBandName'),
                dcc.Dropdown(id='agiBandName',
                    options=[{'label': i, 'value': i} for i in ['<0','0-10K','10k-20K','20k-30K','30k-40K','40k-50K','50-60K','60-70K','70-80K','80-90K','90-100K','100K+']],
                    value='30k-40K'),
                html.Div('priorYearAgiBandName'),
                dcc.Dropdown(id='priorYearAgiBandName',
                    options=[{'label': i, 'value': i} for i in ['<0','0-10K','10k-20K','20k-30K','30k-40K','40k-50K','50-60K','60-70K','70-80K','80-90K','90-100K','100K+']],
                    value='30k-40K'),
                html.Div('isRenter'),
                dcc.Input(id='isRenter', value=0, type='number', min=0, max=1, step=1),
                html.Div('priorYearIsRenter'),
                dcc.Input(id='priorYearIsRenter', value=0, type='number', min=0, max=1, step=1),
                html.Div('lostIncome'),
                dcc.Input(id='lostIncome', value=0, type='number', min=0, max=1, step=1),
                html.Div('filingStatusName'),
                dcc.Dropdown(id='filingStatusName',
                    options=[{'label': i, 'value': i} for i in ['Single','HeadOfHousehold','MarriedFilingJointly','MarriedFilingSeparately','QualifyingWidower']],
                    value='Single'),
                html.Div('priorYearFilingStatusName'),
                dcc.Dropdown(id='priorYearFilingStatusName',
                    options=[{'label': i, 'value': i} for i in ['Single','HeadOfHousehold','MarriedFilingJointly','MarriedFilingSeparately','QualifyingWidower']],
                    value='Single'),
                html.Div('priorYearCompleteSku'),
                dcc.Dropdown(id='priorYearCompleteSku',
                    options=[{'label': i, 'value': i} for i in ['100|FFA','200|Free TTO','600|Paid Deluxe','800|Paid Premier','850|Paid Self Employed','910|Paid TTL Basic','920|Paid TTL Deluxe','930|Paid TTL Premier','940|Paid TTL SE','960|Paid TTLF Basic','965|Paid TTLF Deluxe','970|Paid TTLF Premier','975|Paid TTLF SE','NotCompleted']],
                    value='NotCompleted'),
                html.Div('priorYearStartMonth'),
                dcc.Dropdown(id='priorYearStartMonth',
                    options=[{'label': i, 'value': i} for i in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']],
                    value='Jan'),

            ], className='two columns'),
            html.Div([
                html.H3('Predictions'),
                html.Div('Predicted Status:'),
                html.Div(id='PredResults'),
            ], className='two columns')
        ], className='twelve columns',
    ),

    html.Br(),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl),
    ]
)

######### Define Callback: Predictions
@app.callback(
    [Output(component_id='PredResults', component_property='children'),
    ],
    [Input(component_id='fiveYearTenureName', component_property='value'),
     Input(component_id='basic_segment_type_name', component_property='value'),
     Input(component_id='ageInYears', component_property='value'),
     Input(component_id='agiBandName', component_property='value'),
     Input(component_id='priorYearAgiBandName', component_property='value'),
     Input(component_id='isRenter', component_property='value'),
     Input(component_id='priorYearIsRenter', component_property='value'),
     Input(component_id='lostIncome', component_property='value'),
     Input(component_id='filingStatusName', component_property='value'),
     Input(component_id='priorYearFilingStatusName', component_property='value'),
     Input(component_id='priorYearCompleteSku', component_property='value'),
     Input(component_id='priorYearStartMonth', component_property='value')
    ])
def func(*args):
    listofargs=[arg for arg in args[:11]]
    return make_predictions(listofargs, args[11])

############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
