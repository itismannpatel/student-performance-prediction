"""
Data preprocessing functions for Student Performance Prediction System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import config
import utils

def clean_data(df):
    """Clean and validate the dataset"""
    df_clean = df.copy()
    
    # Remove any duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle missing values (if any)
    df_clean = df_clean.dropna()
    
    # Ensure categorical variables are in correct format
    for col in config.CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    # Ensure numeric variables are numeric
    for col in config.NUMERIC_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Ensure target is numeric
    df_clean[config.TARGET] = pd.to_numeric(df_clean[config.TARGET], errors='coerce')
    
    # Remove any rows with NaN after conversion
    df_clean = df_clean.dropna()
    
    return df_clean

def encode_categorical_features(df, encoder_dict=None, fit=True):
    """
    Encode categorical features using label encoding
    Returns: encoded dataframe and encoder dictionary
    """
    df_encoded = df.copy()
    
    if fit:
        encoder_dict = {}
        for col in config.CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoder_dict[col] = le
    else:
        if encoder_dict is None:
            raise ValueError("encoder_dict must be provided when fit=False")
        for col in config.CATEGORICAL_FEATURES:
            if col in df_encoded.columns and col in encoder_dict:
                le = encoder_dict[col]
                # Handle unseen labels
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    return df_encoded, encoder_dict

def scale_features(X, scaler=None, fit=True):
    """
    Scale numeric features using StandardScaler
    Returns: scaled features and scaler object
    """
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler

def prepare_features(df, include_engineered=True):
    """
    Prepare features for modeling
    Returns: feature matrix (X) and target vector (y)
    """
    df_prep = df.copy()
    
    # Add engineered features if requested
    if include_engineered:
        df_prep = utils.calculate_engineered_features(df_prep)
    
    # Separate features and target
    if config.TARGET in df_prep.columns:
        y = df_prep[config.TARGET]
        X = df_prep.drop(columns=[config.TARGET])
    else:
        y = None
        X = df_prep
    
    return X, y

def split_data(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """Split data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

def preprocess_single_input(input_dict, encoder_dict, scaler, feature_names):
    """
    Preprocess a single input for prediction
    
    Args:
        input_dict: Dictionary with feature values
        encoder_dict: Dictionary of label encoders
        scaler: StandardScaler object
        feature_names: List of feature names in correct order
    
    Returns:
        Preprocessed feature array ready for prediction
    """
    # Create dataframe from input
    df_input = pd.DataFrame([input_dict])
    
    # Add engineered features
    df_input = utils.calculate_engineered_features(df_input)
    
    # Encode categorical features
    df_encoded, _ = encode_categorical_features(df_input, encoder_dict=encoder_dict, fit=False)
    
    # Ensure all features are present and in correct order
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    df_encoded = df_encoded[feature_names]
    
    # Scale features
    X_scaled, _ = scale_features(df_encoded, scaler=scaler, fit=False)
    
    return X_scaled

def get_feature_statistics(df):
    """Calculate statistics for each feature"""
    stats = {}
    
    for col in df.columns:
        if col in config.NUMERIC_FEATURES or col == config.TARGET:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75)
            }
        elif col in config.CATEGORICAL_FEATURES:
            stats[col] = {
                'unique': df[col].nunique(),
                'mode': df[col].mode()[0] if not df[col].mode().empty else None,
                'top_values': df[col].value_counts().head(3).to_dict()
            }
    
    return stats

def validate_input(input_dict):
    """
    Validate user input values
    Returns: (is_valid, error_message)
    """
    errors = []
    
    # Check age range
    if not (15 <= input_dict.get('age', 0) <= 22):
        errors.append("Age must be between 15 and 22")
    
    # Check education levels
    if not (0 <= input_dict.get('Medu', 0) <= 4):
        errors.append("Mother's education must be between 0 and 4")
    if not (0 <= input_dict.get('Fedu', 0) <= 4):
        errors.append("Father's education must be between 0 and 4")
    
    # Check scale variables (1-5)
    scale_vars = ['traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
    for var in scale_vars:
        if not (1 <= input_dict.get(var, 0) <= 5):
            errors.append(f"{var} must be between 1 and 5")
    
    # Check failures
    if not (0 <= input_dict.get('failures', 0) <= 4):
        errors.append("Failures must be between 0 and 4")
    
    # Check absences
    if input_dict.get('absences', 0) < 0:
        errors.append("Absences cannot be negative")
    
    # Check grades
    if not (0 <= input_dict.get('G1', 0) <= 20):
        errors.append("G1 grade must be between 0 and 20")
    if not (0 <= input_dict.get('G2', 0) <= 20):
        errors.append("G2 grade must be between 0 and 20")
    
    if errors:
        return False, " | ".join(errors)
    
    return True, "Input is valid"

def prepare_full_pipeline(df):
    """
    Complete preprocessing pipeline for training
    Returns: X_train, X_test, y_train, y_test, scaler, encoder_dict, feature_names
    """
    # Clean data
    df_clean = clean_data(df)
    
    # Prepare features
    X, y = prepare_features(df_clean, include_engineered=True)
    
    # Encode categorical features
    X_encoded, encoder_dict = encode_categorical_features(X, fit=True)
    
    # Get feature names before scaling
    feature_names = X_encoded.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)
    
    # Scale features
    X_train_scaled, scaler = scale_features(X_train, fit=True)
    X_test_scaled, _ = scale_features(X_test, scaler=scaler, fit=False)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder_dict, feature_names