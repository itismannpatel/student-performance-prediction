"""
Batch Predictions Page
"""

import streamlit as st
import sys
import os
import pandas as pd
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils
import model as ml_model
import preprocessing

st.set_page_config(**config.PAGE_CONFIG)

st.title("📑 Batch Predictions")

# Check if model exists
model_exists = (
    os.path.exists(config.MODEL_PATH) and
    os.path.exists(config.SCALER_PATH) and
    os.path.exists(config.ENCODER_PATH) and
    os.path.exists(config.FEATURE_NAMES_PATH)
)

if not model_exists:
    st.warning("⚠️ Please train the model first!")
    st.stop()

# Load model
model, scaler, encoder_dict, feature_names, metadata = ml_model.load_model_artifacts()

st.info("""
📤 Upload a CSV file containing student data to make predictions for multiple students at once.
The file should contain all required features in the same format as the training data.
""")

st.markdown("---")

# Option 1: Use existing dataset
st.markdown("### Option 1: Use Existing Dataset")

if os.path.exists(config.DATA_PATH):
    if st.button("🔮 Predict on Entire Dataset"):
        with st.spinner("Making predictions..."):
            df = utils.load_data()
            
            # Make predictions
            predictions = ml_model.predict_batch(df, model, scaler, encoder_dict, feature_names)
            
            # Add predictions to dataframe
            df['Predicted_G3'] = predictions
            df['Prediction_Error'] = df['Predicted_G3'] - df['G3']
            df['Grade_Category'] = df['Predicted_G3'].apply(utils.get_grade_category)
            df['Risk_Level'] = df['Predicted_G3'].apply(utils.get_risk_level)
            
            st.success(f"✅ Predictions complete for {len(df)} students!")
            
            # Display summary
            st.markdown("### 📊 Prediction Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mae = abs(df['Prediction_Error']).mean()
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            
            with col2:
                rmse = (df['Prediction_Error'] ** 2).mean() ** 0.5
                st.metric("RMSE", f"{rmse:.2f}")
            
            with col3:
                within_1 = (abs(df['Prediction_Error']) <= 1).sum()
                accuracy = within_1 / len(df) * 100
                st.metric("Accuracy (±1)", f"{accuracy:.1f}%")
            
            with col4:
                high_risk = (df['Risk_Level'] == 'High Risk').sum()
                st.metric("High Risk Students", high_risk)
            
            st.markdown("---")
            
            # Display results
            st.markdown("### 📋 Prediction Results")
            
            # Create display dataframe
            display_cols = ['age', 'sex', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3', 
                          'Predicted_G3', 'Prediction_Error', 'Grade_Category', 'Risk_Level']
            display_df = df[[col for col in display_cols if col in df.columns]]
            
            st.dataframe(
                display_df.style.background_gradient(subset=['Predicted_G3'], cmap='RdYlGn', vmin=0, vmax=20),
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Download results
            st.markdown("### 💾 Download Results")
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
            
            # Visualization
            st.markdown("---")
            st.markdown("### 📊 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                import plotly.express as px
                fig = px.scatter(
                    df,
                    x='G3',
                    y='Predicted_G3',
                    title='Actual vs Predicted Grades',
                    labels={'G3': 'Actual Grade', 'Predicted_G3': 'Predicted Grade'},
                    color='Prediction_Error',
                    color_continuous_scale='RdYlGn_r'
                )
                # Add diagonal line
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=20, y1=20,
                    line=dict(color="red", dash="dash")
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk distribution
                risk_counts = df['Risk_Level'].value_counts()
                fig2 = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='Risk Level Distribution',
                    color=risk_counts.index,
                    color_discrete_map={
                        'High Risk': '#d62728',
                        'Medium Risk': '#ff7f0e',
                        'Low Risk': '#2ca02c'
                    }
                )
                st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Option 2: Upload custom file
st.markdown("### Option 2: Upload Custom CSV File")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file with student data"
)

if uploaded_file is not None:
    try:
        # Read uploaded file
        df_upload = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! ({len(df_upload)} rows)")
        
        # Show preview
        with st.expander("📄 Preview Uploaded Data"):
            st.dataframe(df_upload.head(10))
        
        # Validate columns
        required_features = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
        missing_features = [f for f in required_features if f not in df_upload.columns and f != 'G3']
        
        if missing_features:
            st.warning(f"⚠️ Missing required features: {', '.join(missing_features)}")
            st.info("The system will attempt to use default values for missing features.")
        
        st.markdown("---")
        
        if st.button("🔮 Make Predictions on Uploaded Data"):
            with st.spinner("Processing and making predictions..."):
                try:
                    # Make predictions
                    predictions = ml_model.predict_batch(df_upload, model, scaler, encoder_dict, feature_names)
                    
                    # Add predictions to dataframe
                    df_upload['Predicted_G3'] = predictions
                    df_upload['Grade_Category'] = df_upload['Predicted_G3'].apply(utils.get_grade_category)
                    df_upload['Risk_Level'] = df_upload['Predicted_G3'].apply(utils.get_risk_level)
                    df_upload['Risk_Score'] = df_upload['Predicted_G3'].apply(utils.calculate_risk_score)
                    
                    st.success(f"✅ Predictions complete for {len(df_upload)} students!")
                    
                    # Display summary
                    st.markdown("### 📊 Prediction Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_pred = df_upload['Predicted_G3'].mean()
                        st.metric("Average Predicted Grade", f"{avg_pred:.2f}")
                    
                    with col2:
                        high_performers = (df_upload['Predicted_G3'] >= 16).sum()
                        st.metric("Excellent Students", high_performers)
                    
                    with col3:
                        medium_risk = (df_upload['Risk_Level'] == 'Medium Risk').sum()
                        st.metric("Medium Risk", medium_risk)
                    
                    with col4:
                        high_risk = (df_upload['Risk_Level'] == 'High Risk').sum()
                        st.metric("High Risk", high_risk)
                    
                    st.markdown("---")
                    
                    # Display results
                    st.markdown("### 📋 Prediction Results")
                    st.dataframe(
                        df_upload.style.background_gradient(subset=['Predicted_G3'], cmap='RdYlGn', vmin=0, vmax=20),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    
                    # Download results
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="custom_batch_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # At-risk students list
                    st.markdown("---")
                    st.markdown("### 🚨 At-Risk Students")
                    
                    at_risk_students = df_upload[df_upload['Risk_Level'] == 'High Risk']
                    
                    if len(at_risk_students) > 0:
                        st.warning(f"Found {len(at_risk_students)} students at high risk of failure")
                        
                        st.dataframe(
                            at_risk_students[['Predicted_G3', 'Grade_Category', 'Risk_Score']],
                            use_container_width=True
                        )
                        
                        # Download at-risk list
                        at_risk_csv = at_risk_students.to_csv(index=False)
                        st.download_button(
                            label="📥 Download At-Risk Students List",
                            data=at_risk_csv,
                            file_name="at_risk_students.csv",
                            mime="text/csv"
                        )
                    else:
                        st.success("✅ No high-risk students found!")
                    
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

st.markdown("---")

# Template download
st.markdown("### 📋 Download Template")

st.info("""
💡 **Need a template?** Download a sample CSV template with all required columns 
to ensure your file is formatted correctly.
""")

if st.button("📥 Download CSV Template"):
    # Create template dataframe
    template_data = {
        'school': ['GP'],
        'sex': ['M'],
        'age': [17],
        'address': ['U'],
        'famsize': ['GT3'],
        'Pstatus': ['T'],
        'Medu': [2],
        'Fedu': [2],
        'Mjob': ['other'],
        'Fjob': ['other'],
        'reason': ['course'],
        'guardian': ['mother'],
        'traveltime': [2],
        'studytime': [2],
        'failures': [0],
        'schoolsup': ['no'],
        'famsup': ['yes'],
        'paid': ['no'],
        'activities': ['no'],
        'nursery': ['yes'],
        'higher': ['yes'],
        'internet': ['yes'],
        'romantic': ['no'],
        'famrel': [4],
        'freetime': [3],
        'goout': [3],
        'Dalc': [1],
        'Walc': [1],
        'health': [4],
        'absences': [0],
        'G1': [10],
        'G2': [10]
    }
    
    template_df = pd.DataFrame(template_data)
    template_csv = template_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Template CSV",
        data=template_csv,
        file_name="student_data_template.csv",
        mime="text/csv"
    )
    
    st.success("✅ Template ready for download!")