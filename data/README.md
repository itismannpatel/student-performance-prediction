# Data Folder

This folder contains the dataset required for training and predictions.

## Required File

**student-mat.csv** - UCI Student Performance Dataset (Mathematics course)

⚠️ **Important:** The file MUST be named exactly `student-mat.csv` (or you can use `student-por.csv` for Portuguese course, but update config.py accordingly)

### File Format Check

The CSV file should have these characteristics:
- **Delimiter:** Semicolon (`;`) or comma (`,`)
- **Headers:** First row contains column names
- **Required columns:** Must include G1, G2, G3 (grades)
- **Encoding:** UTF-8

### How to Get the Dataset

1. **Kaggle (Recommended)**
   - Go to: https://www.kaggle.com/datasets
   - Search: "UCI Student Performance Dataset" or "student performance"
   - Download the dataset
   - Extract and place `student-mat.csv` here

2. **UCI Repository**
   - Visit: https://archive.ics.uci.edu/ml/datasets/Student+Performance
   - Download the dataset
   - Extract `student-mat.csv` to this folder

## File Structure

```
data/
├── README.md (this file)
└── student-mat.csv (download separately)
```

## Dataset Information

- **Students:** 395
- **Features:** 33 attributes
- **Target:** G3 (final grade, 0-20 scale)
- **Format:** CSV with headers
- **Size:** ~50KB

## Required Columns

The dataset must contain these columns:
- school, sex, age, address, famsize, Pstatus
- Medu, Fedu, Mjob, Fjob, reason, guardian
- traveltime, studytime, failures
- schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
- famrel, freetime, goout, Dalc, Walc, health, absences
- G1, G2, G3

## Notes

- Do not modify the dataset file name
- Keep the original CSV format with headers
- The G3 column is the target variable
- G1 and G2 are previous grades used as features