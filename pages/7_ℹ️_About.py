"""
About Page
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

st.set_page_config(**config.PAGE_CONFIG)

st.title("ℹ️ About This System")

st.markdown("""
## 📚 Student Performance Prediction System

This application uses machine learning to predict student academic performance based on 
various demographic, social, and academic factors. It was built to help educators identify 
at-risk students and provide targeted interventions.
""")

st.markdown("---")

st.markdown("### 🎯 Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Prediction Capabilities
    - **Single Student Predictions** - Input individual student data
    - **Batch Predictions** - Upload CSV files for multiple students
    - **Confidence Intervals** - Understand prediction uncertainty
    - **Risk Assessment** - Identify students needing intervention
    """)

with col2:
    st.markdown("""
    #### Analysis Tools
    - **What-If Analysis** - Explore impact of changing factors
    - **Feature Importance** - Understand what drives performance
    - **Analytics Dashboard** - Comprehensive data insights
    - **Personalized Recommendations** - Actionable improvement suggestions
    """)

st.markdown("---")

st.markdown("### 📊 Dataset Information")

st.info("""
**UCI Student Performance Dataset**

This system uses the Student Performance dataset from the UCI Machine Learning Repository. 
The data was collected from two Portuguese secondary schools and includes:

- **Students:** 395 from mathematics course
- **Features:** 33 attributes including demographic, social, and school-related information
- **Target:** Final grade (G3) on a scale of 0-20
- **Source:** P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. 2008.
""")

st.markdown("#### Feature Categories")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Demographics", "🏠 Family", "📚 Academic", "🎯 Lifestyle"])

with tab1:
    st.markdown("""
    - **school** - Student's school (GP or MS)
    - **sex** - Student's gender
    - **age** - Student's age (15 to 22)
    - **address** - Home address type (Urban or Rural)
    - **famsize** - Family size
    - **Pstatus** - Parent's cohabitation status
    """)

with tab2:
    st.markdown("""
    - **Medu** - Mother's education (0-4)
    - **Fedu** - Father's education (0-4)
    - **Mjob** - Mother's job
    - **Fjob** - Father's job
    - **guardian** - Student's guardian
    - **famrel** - Quality of family relationships (1-5)
    - **famsup** - Family educational support
    """)

with tab3:
    st.markdown("""
    - **studytime** - Weekly study time (1-4)
    - **failures** - Number of past class failures
    - **schoolsup** - Extra educational support
    - **paid** - Extra paid classes
    - **activities** - Extra-curricular activities
    - **higher** - Wants to take higher education
    - **absences** - Number of school absences
    - **G1** - First period grade
    - **G2** - Second period grade
    """)

with tab4:
    st.markdown("""
    - **traveltime** - Home to school travel time (1-4)
    - **freetime** - Free time after school (1-5)
    - **goout** - Going out with friends (1-5)
    - **Dalc** - Workday alcohol consumption (1-5)
    - **Walc** - Weekend alcohol consumption (1-5)
    - **health** - Current health status (1-5)
    - **internet** - Internet access at home
    - **romantic** - In a romantic relationship
    """)

st.markdown("---")

st.markdown("### 🤖 Machine Learning Models")

st.markdown("""
The system trains and evaluates multiple machine learning algorithms to find the best performer:

1. **Random Forest Regressor**
   - Ensemble of decision trees
   - Handles non-linear relationships well
   - Provides feature importance

2. **Gradient Boosting Regressor**
   - Sequential ensemble method
   - Often achieves high accuracy
   - Good for complex patterns

3. **Linear Regression**
   - Simple baseline model
   - Fast and interpretable
   - Works well for linear relationships

4. **Ridge Regression**
   - Regularized linear regression
   - Prevents overfitting
   - Handles multicollinearity

5. **Support Vector Regression (SVR)**
   - Kernel-based approach
   - Can capture non-linear patterns
   - Robust to outliers

The system automatically selects the best performing model based on RMSE (Root Mean Squared Error).
""")

st.markdown("---")

st.markdown("### 📈 Evaluation Metrics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### MAE (Mean Absolute Error)
    Average absolute difference between predicted and actual grades. 
    Lower is better. Example: MAE of 1.5 means predictions are off by 1.5 points on average.
    
    #### RMSE (Root Mean Squared Error)
    Square root of average squared errors. Penalizes larger errors more than MAE. 
    Lower is better.
    """)

with col2:
    st.markdown("""
    #### R² Score
    Proportion of variance in grades explained by the model. 
    Ranges from 0 to 1, higher is better. 0.8 means 80% of variance is explained.
    
    #### Accuracy (±1 point)
    Percentage of predictions within 1 point of actual grade. 
    Practical measure of prediction usefulness.
    """)

st.markdown("---")

st.markdown("### 🔧 Technical Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### Core Libraries
    - Python 3.8+
    - Streamlit
    - Pandas
    - NumPy
    """)

with col2:
    st.markdown("""
    #### Machine Learning
    - Scikit-learn
    - SHAP (Explainability)
    - Joblib (Model persistence)
    """)

with col3:
    st.markdown("""
    #### Visualization
    - Plotly
    - Matplotlib
    - Seaborn
    """)

st.markdown("---")

st.markdown("### 💡 How It Works")

st.markdown("""
1. **Data Preprocessing**
   - Clean and validate input data
   - Encode categorical variables
   - Scale numeric features
   - Engineer additional features

2. **Model Training**
   - Train multiple algorithms
   - Perform cross-validation
   - Select best performer
   - Save model artifacts

3. **Prediction**
   - Load saved model
   - Preprocess new data
   - Make predictions
   - Calculate confidence intervals

4. **Analysis**
   - Generate recommendations
   - Identify risk factors
   - Visualize patterns
   - Explain predictions
""")

st.markdown("---")

st.markdown("### 🎓 Use Cases")

st.markdown("""
- **Early Warning System** - Identify students at risk of failure early
- **Intervention Planning** - Prioritize resources for at-risk students
- **Performance Monitoring** - Track student progress over time
- **Policy Making** - Understand factors affecting student success
- **Resource Allocation** - Optimize support system deployment
- **Educational Research** - Study patterns in student performance
""")

st.markdown("---")

st.markdown("### 📝 Limitations & Considerations")

st.warning("""
**Important Notes:**

- Predictions are probabilistic, not deterministic
- Model accuracy depends on training data quality
- Should be used as a supplementary tool, not sole decision maker
- Requires regular retraining with new data
- May not capture all factors affecting student performance
- Ethical considerations in using AI for educational decisions
""")

st.markdown("---")

st.markdown("### 📚 References")

st.markdown("""
1. P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance." 
   In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) 
   pp. 5-12, Porto, Portugal, April, 2008.

2. UCI Machine Learning Repository: Student Performance Data Set
   https://archive.ics.uci.edu/ml/datasets/Student+Performance

3. Scikit-learn Documentation
   https://scikit-learn.org/

4. Streamlit Documentation
   https://docs.streamlit.io/
""")

st.markdown("---")

st.markdown("### 🤝 Support & Feedback")

st.info("""
**Questions or Issues?**

This is an educational project demonstrating machine learning for student performance prediction.
For questions, suggestions, or bug reports, please refer to the project documentation.

**Version:** 1.0.0  
**Last Updated:** 2024
""")

st.markdown("---")

st.success("""
✨ **Thank you for using the Student Performance Prediction System!**

We hope this tool helps educators make data-driven decisions to support student success.
""")