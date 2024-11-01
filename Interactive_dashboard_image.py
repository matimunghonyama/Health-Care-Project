import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

def load_data():
    file_path = r"C:\Users\eKasi_SWT_COM00862\Desktop\Health_Care_Project\Health_Care_Services.csv"
    df = pd.read_csv(file_path)
    
    df['Date Issued'] = pd.to_datetime(df['Date Issued'], errors='coerce', dayfirst=True)
    
    if df['Date Issued'].isna().any():
        print("Some dates could not be parsed:")
        print(df[df['Date Issued'].isna()]) 
    return df


df = load_data()

app.layout = html.Div([
    html.H1("Healthcare Services Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    html.Div([
        html.Div([html.H4("Total Records"), html.H2(id='total-records')], 
                 className='stat-box', style={'width': '23%', 'display': 'inline-block',
                 'textAlign': 'center', 'backgroundColor': '#ecf0f1', 'padding': '20px', 
                 'borderRadius': '5px'}),
        
        html.Div([html.H4("Unique Drugs"), html.H2(id='unique-drugs')], 
                 className='stat-box', style={'width': '23%', 'display': 'inline-block',
                 'textAlign': 'center', 'backgroundColor': '#ecf0f1', 'padding': '20px', 
                 'borderRadius': '5px', 'marginLeft': '2%'}),
        
        html.Div([html.H4("Unique Prescribers"), html.H2(id='unique-prescribers')], 
                 className='stat-box', style={'width': '23%', 'display': 'inline-block',
                 'textAlign': 'center', 'backgroundColor': '#ecf0f1', 'padding': '20px', 
                 'borderRadius': '5px', 'marginLeft': '2%'}),
        
        html.Div([html.H4("Date Range"), html.H2(id='date-range')], 
                 className='stat-box', style={'width': '23%', 'display': 'inline-block',
                 'textAlign': 'center', 'backgroundColor': '#ecf0f1', 'padding': '20px', 
                 'borderRadius': '5px', 'marginLeft': '2%'})
    ], style={'marginBottom': 30}),
    
    html.Div([
        html.Div([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df['Date Issued'].min(),
                max_date_allowed=df['Date Issued'].max(),
                start_date=df['Date Issued'].min(),
                end_date=df['Date Issued'].max()
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Select Drug:"),
            dcc.Dropdown(
                id='drug-dropdown',
                options=[{'label': drug, 'value': drug} for drug in df['Drug Name'].dropna().unique()],
                multi=True
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ], style={'marginBottom': 30}),
    
    html.Div([
        html.Div([dcc.Graph(id='time-series-chart')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='drug-distribution')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ]),
    
    html.Div([
        html.Div([dcc.Graph(id='gender-distribution')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='age-distribution')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ])
])

@app.callback(
    [Output('total-records', 'children'),
     Output('unique-drugs', 'children'),
     Output('unique-prescribers', 'children'),
     Output('date-range', 'children'),
     Output('time-series-chart', 'figure'),
     Output('drug-distribution', 'figure'),
     Output('gender-distribution', 'figure'),
     Output('age-distribution', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('drug-dropdown', 'value')]
)
def update_dashboard(start_date, end_date, selected_drugs):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Date Issued'] >= start_date) &
            (filtered_df['Date Issued'] <= end_date)
        ]
    
    if selected_drugs:
        filtered_df = filtered_df[filtered_df['Drug Name'].isin(selected_drugs)]

    total_records = len(filtered_df)
    unique_drugs = filtered_df['Drug Name'].nunique()
    unique_prescribers = filtered_df['Prescriber Name'].nunique()
    
    if not filtered_df.empty:
        date_range = f"{filtered_df['Date Issued'].min().strftime('%Y-%m-%d')} to {filtered_df['Date Issued'].max().strftime('%Y-%m-%d')}"
    else:
        date_range = "No records found for the selected range"

    time_series = px.line(
        filtered_df.groupby(filtered_df['Date Issued'].dt.date).size().reset_index(),
        x='Date Issued',
        y=0,
        title='Prescriptions Over Time'
    )

    drug_dist = px.pie(
        filtered_df['Drug Name'].value_counts().reset_index(),
        values='count',
        names='Drug Name',
        title='Drug Distribution'
    )

    gender_dist = px.pie(
        filtered_df['sex'].value_counts().reset_index(),
        values='count',
        title='Gender Distribution'
    )

    age_dist = px.histogram(
        filtered_df,
        x='Age',
        title='Age Distribution'
    )
    
    return (
        total_records,
        unique_drugs,
        unique_prescribers,
        date_range,
        time_series,
        drug_dist,
        gender_dist,
        age_dist
    )

if __name__ == '__main__':
    app.run_server(debug=True)
