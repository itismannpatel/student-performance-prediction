# Quick Setup Guide

## Folder Structure Setup

After downloading/cloning the project, create the required folders:

```bash
mkdir data
mkdir models
mkdir components
mkdir pages
```

## File Organization

Ensure files are in the correct locations:

```
project-root/
├── app.py                          (main file)
├── config.py
├── utils.py
├── preprocessing.py
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
├── SETUP_GUIDE.md
│
├── components/
│   ├── __init__.py                 (create empty file)
│   ├── charts.py
│   ├── forms.py
│   └── cards.py
│
├── pages/
│   ├── 3_🔍_What_If_Analysis.py
│   ├── 4_📊_Analytics_Dashboard.py
│   ├── 5_📈_Feature_Analysis.py
│   ├── 6_📑_Batch_Predictions.py
│   └── 7_ℹ️_About.py
│
├── data/
│   └── student-mat.csv            (download separately)
│
└── models/
    └── (files generated after training)
```

## Creating Empty __init__.py

For the components folder to work as a Python package:

**Windows (Command Prompt):**
```cmd
type nul > components\__init__.py
```

**Mac/Linux (Terminal):**
```bash
touch components/__init__.py
```

**Or:** Simply create an empty text file named `__init__.py` in the components folder

## Dataset Download Instructions

1. Visit Kaggle: https://www.kaggle.com/datasets
2. Search for "Student Performance Data Set" or "UCI Student Performance"
3. Download the dataset (you may need to create a free Kaggle account)
4. Find the file named `student-mat.csv` in the downloaded files
5. Copy `student-mat.csv` to your `data/` folder

**Alternative sources:**
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Student+Performance
- Direct download from UCI (if available)

## Installation Steps

### Step 1: Verify Python
```bash
python --version
```
Should show Python 3.8 or higher

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
streamlit --version
```

### Step 4: Run Application
```bash
streamlit run app.py
```

## First-Time Usage

1. **Launch the app** - Run `streamlit run app.py`
2. **Train the model** - Go to "Train Model" page and click "Start Training"
3. **Wait for completion** - Training takes 1-2 minutes
4. **Start predicting** - Navigate to "Make Prediction" page

## Common Issues

### Issue: "No module named 'components'"
**Solution:** Create `__init__.py` file in components folder

### Issue: "Dataset not found"
**Solution:** Ensure `student-mat.csv` is in the `data/` folder

### Issue: "Model not trained"
**Solution:** Complete model training before making predictions

### Issue: Import errors
**Solution:** Reinstall requirements: `pip install -r requirements.txt --upgrade`

### Issue: Port already in use
**Solution:** Run with different port: `streamlit run app.py --server.port 8502`

## Testing the Installation

After setup, verify everything works:

1. App launches without errors
2. Home page displays correctly
3. Can navigate to Train Model page
4. Dataset loads successfully
5. Model training completes
6. Can make predictions

## Performance Optimization

- First run may be slower as Streamlit compiles
- Model training is one-time process (unless retraining)
- Predictions are fast after model is trained
- Large batch predictions may take longer

## System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 500MB free disk space

**Recommended:**
- Python 3.9+
- 8GB RAM
- 1GB free disk space
- Modern web browser (Chrome, Firefox, Edge)

## Additional Notes

- Models are saved and persist between sessions
- No database required - all data in CSV format
- Internet connection needed for Kaggle dataset download
- Application runs locally on your machine
- No API keys or authentication required