import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd


accuracy_data = {
    'name': ['Inter-annotator Agreement', 'Medical Terminology', 'Service Classification', 'Provider Identification'],
    'value': [0.92, 0.97, 0.96, 0.99],
    'target': [0.90, 0.98, 0.95, 0.98],
}

completeness_data = {
    'name': ['Required Fields', 'Supporting Info', 'Patient Demographics', 'Geographic Coverage'],
    'value': [100, 94, 92, 100],
    'target': [100, 95, 90, 100],
}

processing_data = {
    'name': ['Annotations/Hour', 'Quality Verification', 'Error Correction (hrs)', 'Documentation'],
    'value': [45, 100, 20, 98],
    'target': [40, 100, 24, 100],
}

trend_data = {
    'month': ['Jan', 'Feb', 'Mar', 'Apr'],
    'accuracy': [0.91, 0.93, 0.94, 0.96],
    'completeness': [95, 96, 97, 98],
    'processing': [38, 41, 43, 45],
}


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Healthcare Annotation KPI Dashboard"),
    dcc.Tabs([
        dcc.Tab(label='Current Metrics', children=[
            html.Div([
                html.H2("Annotation Accuracy"),
                dcc.Graph(
                    figure=go.Figure(
                        data=[
                            go.Indicator(
                                mode="number+gauge+delta",
                                value=accuracy_data['value'][0],
                                title={"text": accuracy_data['name'][0]},
                                gauge={
                                    'axis': {'range': [None, 1]},
                                    'bar': {'color': "#4CAF50"},
                                    'steps': [
                                        {'range': [0, 0.9], 'color': "#FFA726"},
                                        {'range': [0.9, 1], 'color': "#4CAF50"},
                                    ],
                                },
                            )
                        ]
                    )
                ),
                html.H2("Data Completeness"),
                dcc.Graph(
                    figure=go.Figure(data=[
                        go.Bar(name='Current', x=completeness_data['name'], y=completeness_data['value']),
                        go.Bar(name='Target', x=completeness_data['name'], y=completeness_data['target']),
                    ]).update_layout(barmode='group')
                ),
                html.H2("Processing Standards"),
                dcc.Graph(
                    figure=go.Figure(data=[
                        go.Bar(name='Current', x=processing_data['name'], y=processing_data['value']),
                        go.Bar(name='Target', x=processing_data['name'], y=processing_data['target']),
                    ]).update_layout(barmode='group')
                ),
            ]),
        ]),
        dcc.Tab(label='Trends', children=[
            html.Div([
                html.H2("KPI Trends Over Time"),
                dcc.Graph(
                    figure=go.Figure(data=[
                        go.Scatter(x=trend_data['month'], y=trend_data['accuracy'], mode='lines+markers', name='Accuracy'),
                        go.Scatter(x=trend_data['month'], y=trend_data['completeness'], mode='lines+markers', name='Completeness'),
                        go.Scatter(x=trend_data['month'], y=trend_data['processing'], mode='lines+markers', name='Processing Speed'),
                    ]).update_layout(title='KPI Trends Over Time')
                ),
            ]),
        ]),
    ]),
])


if __name__ == '__main__':
    app.run_server(debug=True)
