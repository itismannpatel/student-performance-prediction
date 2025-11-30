"""
Visualization components for Student Performance Prediction System
"""

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config

def create_gauge_chart(value, title="Predicted Grade"):
    """Create a gauge chart for grade prediction"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 12, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 20], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': '#d62728'},
                {'range': [10, 12], 'color': '#ff7f0e'},
                {'range': [12, 14], 'color': '#ffdd57'},
                {'range': [14, 16], 'color': '#2ca02c'},
                {'range': [16, 20], 'color': '#1f77b4'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 10
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_feature_importance_chart(feature_importance_df, top_n=10):
    """Create horizontal bar chart for feature importance"""
    top_features = feature_importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_grade_distribution(df, target_col='G3'):
    """Create histogram of grade distribution"""
    fig = px.histogram(
        df,
        x=target_col,
        nbins=20,
        title='Grade Distribution',
        labels={target_col: 'Final Grade (G3)', 'count': 'Number of Students'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Final Grade",
        yaxis_title="Frequency"
    )
    
    return fig

def create_correlation_heatmap(df, features=None):
    """Create correlation heatmap for numeric features"""
    if features is None:
        features = config.NUMERIC_FEATURES + [config.TARGET]
    
    # Filter to only include numeric columns that exist
    numeric_cols = [col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Feature Correlation Heatmap"
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_radar_chart(features_dict, categories):
    """Create radar chart for student profile"""
    values = [features_dict.get(cat, 0) for cat in categories]
    values += values[:1]  # Complete the circle
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        name='Student Profile',
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=False,
        title="Student Performance Profile",
        height=400
    )
    
    return fig

def create_grade_trend_chart(g1, g2, g3):
    """Create line chart showing grade progression"""
    periods = ['Period 1 (G1)', 'Period 2 (G2)', 'Final (G3)']
    grades = [g1, g2, g3]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=periods,
        y=grades,
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=12),
        name='Grade Progression'
    ))
    
    fig.update_layout(
        title='Grade Progression Over Time',
        xaxis_title='Period',
        yaxis_title='Grade (0-20)',
        yaxis=dict(range=[0, 20]),
        height=350,
        showlegend=False
    )
    
    return fig

def create_comparison_chart(original_value, modified_value, feature_name):
    """Create comparison bar chart for what-if analysis"""
    fig = go.Figure(data=[
        go.Bar(
            name='Original',
            x=['Original', 'Modified'],
            y=[original_value, modified_value],
            marker_color=['#1f77b4', '#2ca02c']
        )
    ])
    
    fig.update_layout(
        title=f'Impact of Changing {feature_name}',
        yaxis_title='Predicted Grade',
        yaxis=dict(range=[0, 20]),
        height=350,
        showlegend=False
    )
    
    return fig

def create_risk_distribution_pie(df):
    """Create pie chart showing distribution of risk levels"""
    risk_counts = df['risk_level'].value_counts()
    
    colors = {
        'High Risk': '#d62728',
        'Medium Risk': '#ff7f0e',
        'Low Risk': '#2ca02c'
    }
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Student Risk Level Distribution',
        color=risk_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_box_plot(df, feature, target='G3'):
    """Create box plot comparing feature values across grade ranges"""
    fig = px.box(
        df,
        x=feature,
        y=target,
        title=f'Grade Distribution by {feature}',
        labels={target: 'Final Grade', feature: feature.replace('_', ' ').title()},
        color=feature
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_scatter_plot(df, x_feature, y_feature='G3'):
    """Create scatter plot for two features"""
    try:
        # Try with trendline
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            title=f'{y_feature} vs {x_feature}',
            labels={
                x_feature: x_feature.replace('_', ' ').title(),
                y_feature: 'Final Grade'
            },
            trendline="ols",
            color=y_feature,
            color_continuous_scale='Viridis'
        )
    except Exception:
        # Fallback without trendline if statsmodels not available
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            title=f'{y_feature} vs {x_feature}',
            labels={
                x_feature: x_feature.replace('_', ' ').title(),
                y_feature: 'Final Grade'
            },
            color=y_feature,
            color_continuous_scale='Viridis'
        )
    
    fig.update_layout(height=400)
    
    return fig

def create_multi_model_comparison(evaluations_dict):
    """Create bar chart comparing multiple models"""
    models = list(evaluations_dict.keys())
    rmse_values = [evaluations_dict[model]['RMSE'] for model in models]
    r2_values = [evaluations_dict[model]['R2'] for model in models]
    
    fig = go.Figure(data=[
        go.Bar(name='RMSE', x=models, y=rmse_values, marker_color='#1f77b4'),
        go.Bar(name='R² Score', x=models, y=r2_values, marker_color='#2ca02c')
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=400
    )
    
    return fig

def create_confidence_interval_chart(prediction, lower, upper):
    """Create chart showing prediction with confidence interval"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=['Prediction'],
        y=[prediction],
        mode='markers',
        marker=dict(size=15, color='#1f77b4'),
        name='Predicted Grade',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[upper - prediction],
            arrayminus=[prediction - lower],
            color='#ff7f0e',
            thickness=3,
            width=10
        )
    ))
    
    fig.update_layout(
        title='Prediction with Confidence Interval',
        yaxis_title='Grade (0-20)',
        yaxis=dict(range=[0, 20]),
        height=350,
        showlegend=False
    )
    
    return fig