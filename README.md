# 📚 Student Performance Prediction System

A machine learning application that predicts student academic performance based on demographic, social, and academic factors.

## Features

- **Grade Prediction** - Predict final student grades using trained ML models
- **Risk Assessment** - Identify students at risk of academic failure
- **What-If Analysis** - Simulate impact of changing various factors
- **Batch Processing** - Make predictions for multiple students via CSV upload
- **Analytics Dashboard** - Visualize patterns and trends in student data
- **Feature Analysis** - Understand which factors most influence performance
- **Personalized Recommendations** - Generate actionable improvement suggestions

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project**
```bash
git clone <repository-url>
cd student-performance-prediction-main
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
   - Visit Kaggle and search for "UCI Student Performance Dataset"
   - Download `student-mat.csv`
   - Create a `data` folder in the project root
   - Place the CSV file as `data/student-mat.csv`

4. **Create necessary folders**
```bash
mkdir data 
mkdir models
```

### Project Structure
```
student-performance-prediction/
├── app.py
├── config.py
├── utils.py
├── preprocessing.py
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
├── components/
│   ├── charts.py
│   ├── forms.py
│   └── cards.py
├── pages/
│   ├── 3_🔍_What_If_Analysis.py
│   ├── 4_📊_Analytics_Dashboard.py
│   ├── 5_📈_Feature_Analysis.py
│   ├── 6_📑_Batch_Predictions.py
│   └── 7_ℹ️_About.py
├── data/
│   └── student-mat.csv
└── models/
    └── (generated after training)
```

## Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## Usage Guide

### 1. Train Model
- Navigate to **Train Model** page
- Click "Start Training" button
- Wait for training to complete (1-2 minutes)
- System trains 5 different algorithms and selects the best performer

### 2. Make Predictions
- Go to **Make Prediction** page
- Choose between Complete Form or Quick Prediction
- Fill in student information
- Click "Predict Grade" to see results
- View predicted grade, risk level, and recommendations

### 3. What-If Analysis
- Access **What-If Analysis** page
- Make a baseline prediction first
- Select a feature to modify (study time, absences, etc.)
- Change the value and click "Simulate Impact"
- See how the change affects the predicted grade

### 4. View Analytics
- Open **Analytics Dashboard** page
- Explore dataset statistics and visualizations
- Compare different student groups
- View grade distributions and correlations

### 5. Feature Analysis
- Visit **Feature Analysis** page
- View which features most influence predictions
- Examine feature importance rankings
- Compare model performance metrics

### 6. Batch Predictions
- Go to **Batch Predictions** page
- Option 1: Use existing dataset for bulk predictions
- Option 2: Upload custom CSV file with student data
- Download prediction results as CSV
- View summary statistics and at-risk student lists

## Technical Details

### Machine Learning Models
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression
- Ridge Regression
- Support Vector Regression (SVR)

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Accuracy (within ±1 point)

### Key Technologies
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computations

## Dataset Information

**Source:** UCI Machine Learning Repository - Student Performance Dataset

**Features:** 33 attributes including:
- Demographics (age, gender, address)
- Family background (parent education, jobs)
- Academic history (previous grades, failures, study time)
- Lifestyle factors (health, free time, alcohol consumption)
- Support systems (school support, family support)

**Target:** Final grade (G3) on scale of 0-20

## Troubleshooting

**Model not found error:**
- Train the model first using the Train Model page

**Dataset not found error:**
- Ensure `student-mat.csv` is in the `data/` folder

**Import errors:**
- Verify all packages are installed: `pip install -r requirements.txt`

**Page not loading:**
- Check that Streamlit is running: `streamlit run app.py`

## Notes

- Models are saved in `models/` folder after training
- Predictions are not deterministic - use as supplementary decision tool
- Retrain model periodically with updated data
- CSV uploads for batch predictions must match dataset format

## License

Educational project for demonstration purposes.

## Dataset Citation

P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance." 

Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008.
