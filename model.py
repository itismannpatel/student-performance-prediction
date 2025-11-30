"""
Model training and prediction functions for Student Performance Prediction System
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import config
import utils
import preprocessing

def train_multiple_models(X_train, y_train):
    """Train multiple models and return them with their names"""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=config.RANDOM_STATE
        ),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy (within 1 point)
    accuracy = np.mean(np.abs(y_test - y_pred) <= 1) * 100
    
    return {
        'MAE': round(mae, 3),
        'MSE': round(mse, 3),
        'RMSE': round(rmse, 3),
        'R2': round(r2, 3),
        'Accuracy': round(accuracy, 2)
    }

def cross_validate_model(model, X, y, cv=config.CV_FOLDS):
    """Perform cross-validation"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    return {
        'mean_mae': round(-scores.mean(), 3),
        'std_mae': round(scores.std(), 3)
    }

def select_best_model(trained_models, X_test, y_test):
    """Select the best performing model based on RMSE"""
    best_model = None
    best_name = None
    best_rmse = float('inf')
    
    evaluations = {}
    
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        evaluations[name] = metrics
        
        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            best_model = model
            best_name = name
    
    return best_model, best_name, evaluations

def train_and_save_model(df):
    """
    Complete training pipeline
    Returns: model, metrics, and saves all artifacts
    """
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, encoder_dict, feature_names = \
        preprocessing.prepare_full_pipeline(df)
    
    # Train multiple models
    trained_models = train_multiple_models(X_train, y_train)
    
    # Select best model
    best_model, best_name, all_evaluations = select_best_model(trained_models, X_test, y_test)
    
    # Get best model metrics
    best_metrics = all_evaluations[best_name]
    
    # Cross-validation on best model
    cv_results = cross_validate_model(best_model, X_train, y_train)
    
    # Save model and preprocessing objects
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(encoder_dict, config.ENCODER_PATH)
    joblib.dump(feature_names, config.FEATURE_NAMES_PATH)
    
    # Save metadata
    metadata = {
        'model_type': best_name,
        'metrics': best_metrics,
        'cv_results': cv_results,
        'all_model_evaluations': all_evaluations,
        'feature_count': len(feature_names),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    utils.save_metadata(metadata)
    
    return best_model, best_name, metadata, all_evaluations

def load_model_artifacts():
    """Load all saved model artifacts"""
    try:
        model = joblib.load(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        encoder_dict = joblib.load(config.ENCODER_PATH)
        feature_names = joblib.load(config.FEATURE_NAMES_PATH)
        metadata = utils.load_metadata()
        
        return model, scaler, encoder_dict, feature_names, metadata
    except FileNotFoundError as e:
        return None, None, None, None, None

def predict_single(input_dict, model, scaler, encoder_dict, feature_names):
    """Make prediction for a single student"""
    # Preprocess input
    X_processed = preprocessing.preprocess_single_input(
        input_dict, encoder_dict, scaler, feature_names
    )
    
    # Make prediction
    prediction = model.predict(X_processed)[0]
    
    # Ensure prediction is within valid range
    prediction = np.clip(prediction, 0, 20)
    
    return round(prediction, 2)

def predict_batch(df, model, scaler, encoder_dict, feature_names):
    """Make predictions for multiple students"""
    predictions = []
    
    for idx, row in df.iterrows():
        input_dict = row.to_dict()
        try:
            pred = predict_single(input_dict, model, scaler, encoder_dict, feature_names)
            predictions.append(pred)
        except Exception as e:
            predictions.append(None)
    
    return predictions

def get_feature_importance(model, feature_names):
    """Get feature importance from tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return feature_importance_df
    else:
        return None

def predict_with_confidence(input_dict, model, scaler, encoder_dict, feature_names):
    """
    Make prediction with confidence interval (for ensemble models)
    """
    prediction = predict_single(input_dict, model, scaler, encoder_dict, feature_names)
    
    # For Random Forest, get prediction from all trees
    if hasattr(model, 'estimators_'):
        X_processed = preprocessing.preprocess_single_input(
            input_dict, encoder_dict, scaler, feature_names
        )
        tree_predictions = [tree.predict(X_processed)[0] for tree in model.estimators_]
        
        lower_bound = np.percentile(tree_predictions, 25)
        upper_bound = np.percentile(tree_predictions, 75)
        
        return prediction, round(lower_bound, 2), round(upper_bound, 2)
    
    # For other models, use simple range
    margin = 1.5
    return prediction, round(prediction - margin, 2), round(prediction + margin, 2)

def simulate_what_if(input_dict, model, scaler, encoder_dict, feature_names, 
                     feature_to_change, new_value):
    """
    Simulate what-if scenario by changing a feature value
    """
    # Make copy of input
    modified_input = input_dict.copy()
    
    # Change the specified feature
    modified_input[feature_to_change] = new_value
    
    # Get original prediction
    original_pred = predict_single(input_dict, model, scaler, encoder_dict, feature_names)
    
    # Get new prediction
    new_pred = predict_single(modified_input, model, scaler, encoder_dict, feature_names)
    
    # Calculate impact
    impact = new_pred - original_pred
    
    return {
        'original_prediction': original_pred,
        'new_prediction': new_pred,
        'impact': round(impact, 2),
        'percentage_change': round((impact / original_pred * 100) if original_pred != 0 else 0, 2)
    }