# Models Folder

This folder stores trained machine learning models and preprocessing artifacts.

## Generated Files

After training the model through the application, the following files will be automatically created:

- **trained_model.pkl** - Best performing ML model
- **scaler.pkl** - StandardScaler for feature normalization
- **encoder.pkl** - Label encoders for categorical features
- **feature_names.pkl** - List of feature names in correct order
- **metadata.json** - Model information and performance metrics

## File Structure

```
models/
├── README.md (this file)
├── trained_model.pkl (generated after training)
├── scaler.pkl (generated after training)
├── encoder.pkl (generated after training)
├── feature_names.pkl (generated after training)
└── metadata.json (generated after training)
```

## How to Generate Models

1. Run the application: `streamlit run app.py`
2. Navigate to **Train Model** page
3. Click **Start Training** button
4. Wait for training to complete (1-2 minutes)
5. Models will be saved automatically to this folder

## Model Information

The system trains multiple algorithms:
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression
- Ridge Regression
- Support Vector Regression

The best performing model is automatically selected and saved.

## Notes

- Models persist between application sessions
- Retraining will overwrite existing models
- Do not manually edit .pkl files
- metadata.json contains performance metrics
- All files are required for predictions to work

## Model Artifacts Size

Total size: ~5-10 MB (depending on model complexity)

## Retraining

To retrain the model:
1. Go to Train Model page
2. Click Start Training again
3. New models will replace old ones
4. Previous models are not backed up automatically