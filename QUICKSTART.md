# Quick Start Guide

Get the Student Performance Prediction system running in 5 minutes.

## Step 1: Install Python (if needed)
```bash
python --version
```
Need Python 3.8+. Download from python.org if necessary.

## Step 2: Create Project Folders
```bash
mkdir data 
mkdir models 
mkdir components 
mkdir pages
```

## Step 3: Install Packages
```bash
pip install -r requirements.txt
```

## Step 4: Download Dataset
1. Go to Kaggle.com
2. Search "UCI Student Performance Dataset"
3. Download `student-mat.csv`
4. Put it in the `data/` folder

## Step 5: Create __init__.py
```bash
touch components/__init__.py
```
(Windows: `type nul > components\__init__.py`)

## Step 6: Run the App
```bash
streamlit run app.py
```

## Step 7: Train Model
1. Open http://localhost:8501 in browser
2. Click "Train Model" in sidebar
3. Click "Start Training" button
4. Wait 1-2 minutes

## Step 8: Make Predictions
1. Click "Make Prediction" in sidebar
2. Fill in student information
3. Click "Predict Grade"
4. View results and recommendations

## Done!

You can now:
- Make individual predictions
- Run what-if scenarios
- View analytics
- Process batch predictions

## Troubleshooting

**Port in use?**
```bash
streamlit run app.py --server.port 8502
```

**Missing packages?**
```bash
pip install -r requirements.txt --upgrade
```

**Dataset error?**
Ensure `student-mat.csv` is in `data/` folder

**Import error?**
Create empty `__init__.py` in `components/` folder