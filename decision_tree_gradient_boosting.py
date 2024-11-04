import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df):
    """
    Prepare the healthcare data for modeling with proper class encoding.
    """
    df_model = df.copy()
    
    df_model['Age'] = pd.to_numeric(df_model['Age'].str.extract(r'(\d+)')[0], errors='coerce')
    
    # Process dates
    df_model['Date Issued'] = pd.to_datetime(df_model['Date Issued'], format='mixed', dayfirst=True)
    df_model['Year'] = df_model['Date Issued'].dt.year
    df_model['Month'] = df_model['Date Issued'].dt.month
    
    # Create aspect ratio
    df_model['Aspect_Ratio'] = df_model['Width'] / df_model['Height']
    
    # Select features
    features = ['Width', 'Height', 'Aspect_Ratio', 'Age', 'Year', 'Month']
    X = df_model[features]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Store original class labels before encoding
    unique_classes = df_model['sex'].unique()
    
    # Properly encode target using LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df_model['sex'])
    
    # Get unique classes
    n_classes = len(unique_classes)
    
    return X, y, features, n_classes, unique_classes

def train_and_evaluate_models(X, y, n_classes):
    """
    Train and evaluate both Decision Tree and XGBoost models.
    """
    # Initialize models
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Update XGBoost objective for multiclass
    xgb = XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        objective='multi:softprob',
        num_class=n_classes
    )
    
    # Prepare cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    results = {
        'dt': {'fold_scores': [], 'feature_importance': None},
        'xgb': {'fold_scores': [], 'feature_importance': None}
    }
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Decision Tree
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_val)
        dt_score = {
            'fold': fold,
            'accuracy': accuracy_score(y_val, dt_pred),
            'precision_macro': precision_score(y_val, dt_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_val, dt_pred, average='weighted', zero_division=0),
            'model': 'Decision Tree'
        }
        results['dt']['fold_scores'].append(dt_score)
        
        # XGBoost
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_val)
        xgb_score = {
            'fold': fold,
            'accuracy': accuracy_score(y_val, xgb_pred),
            'precision_macro': precision_score(y_val, xgb_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_val, xgb_pred, average='weighted', zero_division=0),
            'model': 'XGBoost'
        }
        results['xgb']['fold_scores'].append(xgb_score)
    
    # Get feature importance
    results['dt']['feature_importance'] = pd.DataFrame({
        'feature': X.columns,
        'importance': dt.feature_importances_
    })
    
    results['xgb']['feature_importance'] = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb.feature_importances_
    })
    
    return results

# Initialize Dash app
app = dash.Dash(__name__)

# Load and prepare data
df = pd.read_csv("Health_Care_Services.csv") 
X, y, features, n_classes, classes_ = prepare_data(df)
model_results = train_and_evaluate_models(X, y, n_classes)

# Add class information to the dashboard
class_info = html.Div([
    html.H3('Class Information'),
    html.P(f'Number of classes: {n_classes}'),
    html.P(f'Original class labels: {", ".join(str(c) for c in classes_)}')
], style={'marginBottom': 20})

# Combine results for plotting
fold_results = pd.DataFrame(model_results['dt']['fold_scores'] + model_results['xgb']['fold_scores'])

# Create the layout
app.layout = html.Div([
    html.H1('Healthcare Model Analysis Dashboard',
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    class_info,  # Add class information to the dashboard
    
    # Model Comparison Section
    html.Div([
        html.H2('Model Performance Comparison',
                style={'color': '#2c3e50', 'marginBottom': 20}),
        
        # Metric selector
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
        
        # Performance comparison chart
        dcc.Graph(id='model-comparison')
    ], style={'marginBottom': 40}),
    
    # Feature Importance Comparison
    html.Div([
        html.H2('Feature Importance Comparison',
                style={'color': '#2c3e50', 'marginBottom': 20}),
        html.Div([
            html.Div([
                html.H3('Decision Tree Feature Importance',
                        style={'textAlign': 'center'}),
                dcc.Graph(
                    id='dt-feature-importance',
                    figure=px.bar(
                        model_results['dt']['feature_importance'].sort_values('importance', ascending=True),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Decision Tree Feature Importance'
                    ).update_layout(template='plotly_white')
                )
            ], style={'width': '48%'}),
            html.Div([
                html.H3('XGBoost Feature Importance',
                        style={'textAlign': 'center'}),
                dcc.Graph(
                    id='xgb-feature-importance',
                    figure=px.bar(
                        model_results['xgb']['feature_importance'].sort_values('importance', ascending=True),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='XGBoost Feature Importance'
                    ).update_layout(template='plotly_white')
                )
            ], style={'width': '48%'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    ])
])

@app.callback(
    Output('model-comparison', 'figure'),
    [Input('metric-selector', 'value')]
)
def update_model_comparison(selected_metric):
    fig = go.Figure()
    
    for model in ['Decision Tree', 'XGBoost']:
        model_data = fold_results[fold_results['model'] == model]
        
        if selected_metric in ['all', 'accuracy']:
            fig.add_trace(go.Scatter(
                x=model_data['fold'],
                y=model_data['accuracy'],
                name=f'{model} - Accuracy',
                mode='lines+markers'
            ))
        
        if selected_metric in ['all', 'precision']:
            fig.add_trace(go.Scatter(
                x=model_data['fold'],
                y=model_data['precision_macro'],
                name=f'{model} - Precision (Macro)',
                mode='lines+markers'
            ))
            fig.add_trace(go.Scatter(
                x=model_data['fold'],
                y=model_data['precision_weighted'],
                name=f'{model} - Precision (Weighted)',
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title='Model Performance Comparison Across Folds',
        xaxis_title='Fold Number',
        yaxis_title='Score',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)