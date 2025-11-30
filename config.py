"""
Configuration file for Student Performance Prediction System
"""

# File paths
DATA_PATH = "data/student-mat.csv"
PROCESSED_DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/trained_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"
METADATA_PATH = "models/metadata.json"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature definitions
NUMERIC_FEATURES = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
    'failures', 'famrel', 'freetime', 'goout', 'Dalc', 
    'Walc', 'health', 'absences', 'G1', 'G2'
]

CATEGORICAL_FEATURES = [
    'school', 'sex', 'address', 'famsize', 'Pstatus',
    'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
    'famsup', 'paid', 'activities', 'nursery', 'higher',
    'internet', 'romantic'
]

TARGET = 'G3'

# Feature descriptions for UI
FEATURE_DESCRIPTIONS = {
    'school': 'Student\'s school (Gabriel Pereira or Mousinho da Silveira)',
    'sex': 'Student\'s gender',
    'age': 'Student\'s age (15 to 22)',
    'address': 'Home address type (Urban or Rural)',
    'famsize': 'Family size (Less than or equal to 3 or Greater than 3)',
    'Pstatus': 'Parent\'s cohabitation status (Living together or Apart)',
    'Medu': 'Mother\'s education (0=none to 4=higher education)',
    'Fedu': 'Father\'s education (0=none to 4=higher education)',
    'Mjob': 'Mother\'s job',
    'Fjob': 'Father\'s job',
    'reason': 'Reason to choose this school',
    'guardian': 'Student\'s guardian',
    'traveltime': 'Home to school travel time (1=<15min to 4=>1hour)',
    'studytime': 'Weekly study time (1=<2hours to 4=>10hours)',
    'failures': 'Number of past class failures (0 to 4)',
    'schoolsup': 'Extra educational support',
    'famsup': 'Family educational support',
    'paid': 'Extra paid classes',
    'activities': 'Extra-curricular activities',
    'nursery': 'Attended nursery school',
    'higher': 'Wants to take higher education',
    'internet': 'Internet access at home',
    'romantic': 'In a romantic relationship',
    'famrel': 'Quality of family relationships (1=very bad to 5=excellent)',
    'freetime': 'Free time after school (1=very low to 5=very high)',
    'goout': 'Going out with friends (1=very low to 5=very high)',
    'Dalc': 'Workday alcohol consumption (1=very low to 5=very high)',
    'Walc': 'Weekend alcohol consumption (1=very low to 5=very high)',
    'health': 'Current health status (1=very bad to 5=very good)',
    'absences': 'Number of school absences (0 to 93)',
    'G1': 'First period grade (0 to 20)',
    'G2': 'Second period grade (0 to 20)',
    'G3': 'Final grade (0 to 20) - TARGET'
}

# Job categories
JOB_CATEGORIES = ['teacher', 'health', 'services', 'at_home', 'other']

# Reason categories
REASON_CATEGORIES = ['home', 'reputation', 'course', 'other']

# Guardian categories
GUARDIAN_CATEGORIES = ['mother', 'father', 'other']

# Grade thresholds for classification
GRADE_THRESHOLDS = {
    'Excellent': 16,
    'Good': 14,
    'Average': 12,
    'Below Average': 10,
    'Poor': 0
}

# Risk level thresholds
RISK_THRESHOLDS = {
    'High Risk': 10,
    'Medium Risk': 12,
    'Low Risk': 14
}

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'success': '#2ca02c',
    'warning': '#ff7f0e',
    'danger': '#d62728',
    'info': '#17a2b8'
}

# Streamlit page config
PAGE_CONFIG = {
    'page_title': 'Student Performance Predictor',
    'page_icon': '📚',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Model evaluation metrics to track
METRICS = ['MAE', 'MSE', 'RMSE', 'R2', 'Accuracy']