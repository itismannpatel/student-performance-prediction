"""
Main Streamlit Application for Student Performance Prediction System
"""

import streamlit as st
import pandas as pd
import os
import config
import utils
import model as ml_model
import preprocessing

# Page configuration
st.set_page_config(**config.PAGE_CONFIG)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #0d5a8f;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

if 'current_features' not in st.session_state:
    st.session_state.current_features = None

if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

# Sidebar navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🤖 Train Model",
        "🎯 Make Prediction",
        "🔍 What-If Analysis",
        "📊 Analytics Dashboard",
        "📈 Feature Analysis",
        "📑 Batch Predictions",
        "ℹ️ About"
    ]
)

# Check if model exists
model_exists = (
    os.path.exists(config.MODEL_PATH) and
    os.path.exists(config.SCALER_PATH) and
    os.path.exists(config.ENCODER_PATH) and
    os.path.exists(config.FEATURE_NAMES_PATH)
)

# Display model status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")
if model_exists:
    st.sidebar.success("✅ Model Loaded")
    metadata = utils.load_metadata()
    if metadata:
        st.sidebar.info(f"**Algorithm:** {metadata.get('model_type', 'Unknown')}")
        st.sidebar.info(f"**R² Score:** {metadata.get('metrics', {}).get('R2', 0):.3f}")
else:
    st.sidebar.warning("⚠️ No Model Found")
    st.sidebar.info("Please train a model first!")

# Main content based on selected page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">📚 Student Performance Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Academic Success Prediction System</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Predict Grades
        Use machine learning to predict student final grades based on various factors including:
        - Academic history
        - Family background
        - Study habits
        - Lifestyle factors
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Analyze Patterns
        Discover what factors most influence student performance:
        - Feature importance analysis
        - Correlation studies
        - Risk factor identification
        - Performance trends
        """)
    
    with col3:
        st.markdown("""
        ### 💡 Get Recommendations
        Receive personalized, actionable recommendations:
        - Study habit improvements
        - Support system suggestions
        - Risk mitigation strategies
        - Performance optimization
        """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    1. **Train Model** - Load the dataset and train the prediction model
    2. **Make Prediction** - Input student information to get grade predictions
    3. **What-If Analysis** - Explore how changing factors affects predictions
    4. **Analytics** - View comprehensive insights and visualizations
    """)
    
    st.markdown("---")
    
    st.info("""
    📝 **Dataset Information:** This system uses the UCI Student Performance Dataset containing 
    data from Portuguese secondary school students. The dataset includes demographic, social, 
    and academic information used to predict final grades.
    """)
    
    # Display quick stats if data is available
    if os.path.exists(config.DATA_PATH):
        df = utils.load_data()
        if df is not None:
            st.markdown("### 📈 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Students", len(df))
            with col2:
                if 'G3' in df.columns:
                    st.metric("Average Grade", f"{df['G3'].mean():.2f}")
                else:
                    st.metric("Features", len(df.columns))
            with col3:
                st.metric("Total Features", len(df.columns))
            with col4:
                if 'G3' in df.columns:
                    pass_rate = (df['G3'] >= 10).sum() / len(df) * 100
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")
                else:
                    st.metric("Rows", len(df))

elif page == "🤖 Train Model":
    st.title("🤖 Train Prediction Model")
    
    st.info("""
    This page allows you to train the machine learning model using the student performance dataset.
    The system will automatically preprocess the data, train multiple models, and select the best performer.
    """)
    
    # Check if dataset exists
    if not os.path.exists(config.DATA_PATH):
        st.error(f"""
        ❌ Dataset not found at `{config.DATA_PATH}`
        
        Please ensure the dataset file exists in the correct location.
        You can download the UCI Student Performance Dataset from Kaggle.
        """)
    else:
        # Load and display dataset info
        df = utils.load_data()
        
        if df is not None:
            st.success(f"✅ Dataset loaded successfully! ({len(df)} students)")
            
            with st.expander("📊 View Dataset Sample"):
                st.dataframe(df.head(10))
            
            with st.expander("📈 Dataset Statistics"):
                st.write(df.describe())
            
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Training Configuration")
                st.write(f"**Test Size:** {config.TEST_SIZE * 100}%")
                st.write(f"**Cross-Validation Folds:** {config.CV_FOLDS}")
                st.write(f"**Random State:** {config.RANDOM_STATE}")
            
            with col2:
                st.markdown("### Models to Train")
                st.write("- Random Forest")
                st.write("- Gradient Boosting")
                st.write("- Linear Regression")
                st.write("- Ridge Regression")
                st.write("- SVR")
            
            st.markdown("---")
            
            if st.button("🚀 Start Training", key="train_button"):
                with st.spinner("Training models... This may take a minute..."):
                    try:
                        # Create directories if they don't exist
                        os.makedirs("models", exist_ok=True)
                        
                        # Train model
                        best_model, best_name, metadata, all_evaluations = ml_model.train_and_save_model(df)
                        
                        st.session_state.model_trained = True
                        
                        st.success(f"✅ Training complete! Best model: **{best_name}**")
                        
                        # Display results
                        st.markdown("### 📊 Model Performance Comparison")
                        
                        # Create comparison dataframe
                        comparison_df = pd.DataFrame(all_evaluations).T
                        st.dataframe(comparison_df.style.highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE'])
                                   .highlight_max(axis=0, subset=['R2', 'Accuracy']))
                        
                        # Display best model metrics
                        st.markdown(f"### 🏆 Best Model: {best_name}")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        metrics = metadata['metrics']
                        with col1:
                            st.metric("MAE", f"{metrics['MAE']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.3f}")
                        with col3:
                            st.metric("R² Score", f"{metrics['R2']:.3f}")
                        with col4:
                            st.metric("Accuracy", f"{metrics['Accuracy']:.1f}%")
                        with col5:
                            cv = metadata['cv_results']
                            st.metric("CV MAE", f"{cv['mean_mae']:.3f}")
                        
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        st.exception(e)

elif page == "🎯 Make Prediction":
    st.title("🎯 Make Performance Prediction")
    
    if not model_exists:
        st.warning("⚠️ Please train the model first before making predictions!")
        st.stop()
    
    # Load model
    model, scaler, encoder_dict, feature_names, metadata = ml_model.load_model_artifacts()
    
    if model is None:
        st.error("❌ Error loading model artifacts!")
        st.stop()
    
    st.info("Fill in the student information below to predict their final grade.")
    
    # Import forms
    from components import forms, cards, charts
    
    # Form selection
    form_type = st.radio("Choose Input Method:", ["📝 Complete Form", "🚀 Quick Prediction"])
    
    st.markdown("---")
    
    if form_type == "📝 Complete Form":
        form_data = forms.create_student_input_form()
    else:
        form_data = forms.create_quick_prediction_form()
    
    st.markdown("---")
    
    if st.button("🎯 Predict Grade", key="predict_button"):
        # Validate input
        is_valid, error_msg = preprocessing.validate_input(form_data)
        
        if not is_valid:
            st.error(f"❌ Invalid input: {error_msg}")
        else:
            with st.spinner("Making prediction..."):
                try:
                    # Make prediction
                    prediction = ml_model.predict_single(
                        form_data, model, scaler, encoder_dict, feature_names
                    )
                    
                    # Get confidence interval
                    pred, lower, upper = ml_model.predict_with_confidence(
                        form_data, model, scaler, encoder_dict, feature_names
                    )
                    
                    # Store in session state
                    st.session_state.prediction_made = True
                    st.session_state.current_features = form_data
                    st.session_state.current_prediction = prediction
                    
                    # Display results
                    st.markdown("---")
                    
                    # Prediction card
                    cards.display_prediction_card(prediction, form_data)
                    
                    st.markdown("---")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_gauge = charts.create_gauge_chart(prediction)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col2:
                        fig_confidence = charts.create_confidence_interval_chart(pred, lower, upper)
                        st.plotly_chart(fig_confidence, use_container_width=True)
                    
                    # Grade trend
                    if form_data['G1'] and form_data['G2']:
                        st.markdown("---")
                        fig_trend = charts.create_grade_trend_chart(
                            form_data['G1'], form_data['G2'], prediction
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Recommendations
                    recommendations = utils.get_recommendations(form_data, prediction)
                    cards.display_recommendations_card(recommendations)
                    
                    st.success("✅ Prediction complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.exception(e)

else:
    st.info(f"Page '{page}' is under construction. More features coming soon!")
    st.write("This is a placeholder for the selected page.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    <p>📚 Student Performance Prediction System | Built with Streamlit & Scikit-learn</p>
    <p>Dataset: UCI Machine Learning Repository - Student Performance Data Set</p>
</div>
""", unsafe_allow_html=True)