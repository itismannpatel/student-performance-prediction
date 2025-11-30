"""
Utility functions for Student Performance Prediction System
"""

import pandas as pd
import numpy as np
import streamlit as st
import config
from datetime import datetime
import json

def load_data(file_path=config.DATA_PATH):
    """Load the student performance dataset"""
    try:
        data = pd.read_csv(file_path, sep=None, engine='python')
        
        # Check if required columns exist
        required_cols = ['G1', 'G2', 'G3']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            st.error(f"Dataset missing required columns: {', '.join(missing_cols)}")
            st.info("Please ensure you're using the correct dataset file (student-mat.csv or student-por.csv)")
            st.write("Available columns:", list(data.columns))
            return None
        
        return data
    except FileNotFoundError:
        st.error(f"Dataset not found at {file_path}. Please ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_grade_category(grade):
    """Convert numeric grade to category"""
    for category, threshold in config.GRADE_THRESHOLDS.items():
        if grade >= threshold:
            return category
    return 'Poor'

def get_risk_level(grade):
    """Determine risk level based on predicted grade"""
    for risk, threshold in config.RISK_THRESHOLDS.items():
        if grade < threshold:
            return risk
    return 'Low Risk'

def calculate_risk_score(grade, max_grade=20):
    """Calculate risk score from 0-100 (higher = more risk)"""
    return int((1 - (grade / max_grade)) * 100)

def format_percentage(value):
    """Format decimal as percentage"""
    return f"{value * 100:.1f}%"

def create_feature_dict(form_data):
    """Convert form data to feature dictionary for prediction"""
    return {
        'school': form_data['school'],
        'sex': form_data['sex'],
        'age': form_data['age'],
        'address': form_data['address'],
        'famsize': form_data['famsize'],
        'Pstatus': form_data['Pstatus'],
        'Medu': form_data['Medu'],
        'Fedu': form_data['Fedu'],
        'Mjob': form_data['Mjob'],
        'Fjob': form_data['Fjob'],
        'reason': form_data['reason'],
        'guardian': form_data['guardian'],
        'traveltime': form_data['traveltime'],
        'studytime': form_data['studytime'],
        'failures': form_data['failures'],
        'schoolsup': form_data['schoolsup'],
        'famsup': form_data['famsup'],
        'paid': form_data['paid'],
        'activities': form_data['activities'],
        'nursery': form_data['nursery'],
        'higher': form_data['higher'],
        'internet': form_data['internet'],
        'romantic': form_data['romantic'],
        'famrel': form_data['famrel'],
        'freetime': form_data['freetime'],
        'goout': form_data['goout'],
        'Dalc': form_data['Dalc'],
        'Walc': form_data['Walc'],
        'health': form_data['health'],
        'absences': form_data['absences'],
        'G1': form_data['G1'],
        'G2': form_data['G2']
    }

def calculate_engineered_features(df):
    """Create additional engineered features"""
    df_copy = df.copy()
    
    # Grade improvement/decline
    df_copy['grade_trend'] = df_copy['G2'] - df_copy['G1']
    
    # Average parental education
    df_copy['parent_edu_avg'] = (df_copy['Medu'] + df_copy['Fedu']) / 2
    
    # Total alcohol consumption
    df_copy['total_alcohol'] = df_copy['Dalc'] + df_copy['Walc']
    
    # Support system strength (family + school + paid)
    df_copy['support_score'] = (
        df_copy['schoolsup'].map({'yes': 1, 'no': 0}) +
        df_copy['famsup'].map({'yes': 1, 'no': 0}) +
        df_copy['paid'].map({'yes': 1, 'no': 0})
    )
    
    # Risk factors count (failures, high alcohol, many absences, low study time)
    df_copy['risk_factors'] = (
        (df_copy['failures'] > 0).astype(int) +
        (df_copy['total_alcohol'] > 5).astype(int) +
        (df_copy['absences'] > 10).astype(int) +
        (df_copy['studytime'] < 2).astype(int)
    )
    
    # Activity level (activities + going out)
    df_copy['activity_level'] = (
        df_copy['activities'].map({'yes': 1, 'no': 0}) +
        df_copy['goout']
    )
    
    # Home environment score
    df_copy['home_environment'] = (
        df_copy['famrel'] +
        df_copy['internet'].map({'yes': 1, 'no': 0}) +
        (5 - df_copy['traveltime'])  # Less travel time is better
    )
    
    return df_copy

def get_recommendations(features, predicted_grade):
    """Generate personalized recommendations based on features"""
    recommendations = []
    
    # Study time recommendations
    if features.get('studytime', 0) < 3:
        recommendations.append({
            'category': '📚 Study Habits',
            'priority': 'High',
            'suggestion': 'Increase weekly study time to at least 5-10 hours',
            'impact': 'Can improve grade by 1-2 points'
        })
    
    # Attendance recommendations
    if features.get('absences', 0) > 10:
        recommendations.append({
            'category': '🏫 Attendance',
            'priority': 'High',
            'suggestion': 'Reduce absences - aim for <5 absences per period',
            'impact': 'Regular attendance strongly correlates with better grades'
        })
    
    # Failure recovery
    if features.get('failures', 0) > 0:
        recommendations.append({
            'category': '⚠️ Past Performance',
            'priority': 'High',
            'suggestion': 'Seek additional tutoring or educational support',
            'impact': 'Address knowledge gaps from previous failures'
        })
    
    # Support system
    support_count = sum([
        features.get('schoolsup') == 'yes',
        features.get('famsup') == 'yes',
        features.get('paid') == 'yes'
    ])
    if support_count < 2:
        recommendations.append({
            'category': '🤝 Support System',
            'priority': 'Medium',
            'suggestion': 'Consider getting extra educational support (tutoring, family help)',
            'impact': 'Support systems improve grades by 0.5-1 points on average'
        })
    
    # Alcohol consumption
    total_alcohol = features.get('Dalc', 1) + features.get('Walc', 1)
    if total_alcohol > 4:
        recommendations.append({
            'category': '🚫 Lifestyle',
            'priority': 'High',
            'suggestion': 'Reduce alcohol consumption - it negatively impacts academic performance',
            'impact': 'Can improve focus and study effectiveness'
        })
    
    # Social life balance
    if features.get('goout', 0) > 4:
        recommendations.append({
            'category': '⚖️ Balance',
            'priority': 'Medium',
            'suggestion': 'Balance social activities with study time',
            'impact': 'Better time management can lead to improved grades'
        })
    
    # Health
    if features.get('health', 5) < 3:
        recommendations.append({
            'category': '💪 Health',
            'priority': 'Medium',
            'suggestion': 'Focus on improving physical health - it affects academic performance',
            'impact': 'Better health leads to better concentration and energy'
        })
    
    # Higher education aspiration
    if features.get('higher') == 'no':
        recommendations.append({
            'category': '🎯 Goals',
            'priority': 'Low',
            'suggestion': 'Consider exploring higher education options for better career prospects',
            'impact': 'Clear goals increase motivation and performance'
        })
    
    # If no specific issues, give general advice
    if len(recommendations) == 0:
        recommendations.append({
            'category': '✨ General',
            'priority': 'Low',
            'suggestion': 'Maintain your current good habits and stay consistent',
            'impact': 'Consistency is key to sustained success'
        })
    
    return recommendations

def save_metadata(metadata, path=config.METADATA_PATH):
    """Save model metadata to JSON file"""
    metadata['saved_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_metadata(path=config.METADATA_PATH):
    """Load model metadata from JSON file"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def display_metric_card(title, value, delta=None, help_text=None):
    """Display a styled metric card"""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )