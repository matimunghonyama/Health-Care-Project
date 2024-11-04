import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Prepare the data
fold_results = pd.DataFrame({
    'fold': [1, 2, 3, 4, 5],
    'accuracy': [0.65, 0.25, 0.45, 0.45, 0.45],
    'precision0': [0.50, 0.31, 0.50, 0.25, 0.30],
    'precision1': [0.80, 0.14, 0.42, 0.50, 0.60]
})

feature_importance = pd.DataFrame({
    'feature': ['Year', 'Aspect_Ratio', 'Age', 'Width', 'Month', 'Height'],
    'importance': [0.35, 0.25, 0.20, 0.10, 0.05, 0.05]
}).sort_values('importance', ascending=True)

# Create the layout
app.layout = html.Div([
    html.H1('Healthcare Model Analysis Dashboard',
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Model Performance Metrics Section
    html.Div([
        html.H2('Model Performance Metrics',
                style={'color': '#2c3e50', 'marginBottom': 20}),
        
        # Dropdown for metric selection
        html.Div([
            html.Label('Select Metrics to Display:'),
            dcc.Dropdown(
                id='metric-selector',
                options=[
                    {'label': 'All Metrics', 'value': 'all'},
                    {'label': 'Accuracy Only', 'value': 'accuracy'},
                    {'label': 'Precision Metrics', 'value': 'precision'}
                ],
                value='all',
                style={'width': '50%'}
            )
        ], style={'marginBottom': 20}),
        
        # Performance metrics chart
        dcc.Graph(id='performance-metrics')
    ], style={'marginBottom': 40}),
    
    # Feature Importance Section
    html.Div([
        html.H2('Feature Importance Analysis',
                style={'color': '#2c3e50', 'marginBottom': 20}),
        dcc.Graph(
            id='feature-importance',
            figure=go.Figure(
                go.Bar(
                    x=feature_importance['importance'],
                    y=feature_importance['feature'],
                    orientation='h',
                    marker_color='#3498db'
                )
            ).update_layout(
                title='Feature Importance Distribution',
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                template='plotly_white'
            )
        )
    ]),
    
    # Model Summary Statistics
    html.Div([
        html.H2('Model Summary Statistics',
                style={'color': '#2c3e50', 'marginBottom': 20}),
        html.Div([
            html.Div([
                html.H4('Average Performance'),
                html.P(f"Mean Accuracy: {fold_results['accuracy'].mean():.3f}"),
                html.P(f"Std Deviation: {fold_results['accuracy'].std():.3f}")
            ], style={'flex': 1}),
            html.Div([
                html.H4('Best Performance'),
                html.P(f"Best Accuracy: {fold_results['accuracy'].max():.3f}"),
                html.P(f"Best Fold: {fold_results['accuracy'].idxmax() + 1}")
            ], style={'flex': 1}),
            html.Div([
                html.H4('Worst Performance'),
                html.P(f"Worst Accuracy: {fold_results['accuracy'].min():.3f}"),
                html.P(f"Worst Fold: {fold_results['accuracy'].idxmin() + 1}")
            ], style={'flex': 1})
        ], style={'display': 'flex', 'justifyContent': 'space-between'})
    ], style={'marginTop': 40, 'padding': '20px', 'backgroundColor': '#f7f9fc'})
], style={'padding': '20px'})

# Callback to update performance metrics chart
@app.callback(
    Output('performance-metrics', 'figure'),
    [Input('metric-selector', 'value')]
)
def update_performance_chart(selected_metric):
    fig = go.Figure()
    
    if selected_metric in ['all', 'accuracy']:
        fig.add_trace(go.Scatter(
            x=fold_results['fold'],
            y=fold_results['accuracy'],
            name='Accuracy',
            mode='lines+markers',
            line=dict(color='#2ecc71')
        ))
    
    if selected_metric in ['all', 'precision']:
        fig.add_trace(go.Scatter(
            x=fold_results['fold'],
            y=fold_results['precision0'],
            name='Precision (Class 0)',
            mode='lines+markers',
            line=dict(color='#e74c3c')
        ))
        fig.add_trace(go.Scatter(
            x=fold_results['fold'],
            y=fold_results['precision1'],
            name='Precision (Class 1)',
            mode='lines+markers',
            line=dict(color='#3498db')
        ))
    
    fig.update_layout(
        title='Model Performance Across Folds',
        xaxis_title='Fold Number',
        yaxis_title='Score',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)