"""
What-If Analysis Page
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import model as ml_model
from components import forms, cards, charts

st.set_page_config(**config.PAGE_CONFIG)

st.title("🔍 What-If Analysis")

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

# Check if prediction has been made
if not st.session_state.get('prediction_made', False):
    st.info("""
    ℹ️ No prediction found in session. Please make a prediction first from the 
    **Make Prediction** page, or use the quick form below to create a baseline prediction.
    """)
    
    st.markdown("---")
    st.subheader("Quick Baseline Prediction")
    
    form_data = forms.create_quick_prediction_form()
    
    if st.button("Create Baseline"):
        # Load model
        model, scaler, encoder_dict, feature_names, metadata = ml_model.load_model_artifacts()
        
        # Make prediction
        prediction = ml_model.predict_single(
            form_data, model, scaler, encoder_dict, feature_names
        )
        
        # Store in session
        st.session_state.current_features = form_data
        st.session_state.current_prediction = prediction
        st.session_state.prediction_made = True
        
        st.success(f"✅ Baseline prediction created: {prediction:.2f}")
        st.rerun()
    
    st.stop()

# Load model
model, scaler, encoder_dict, feature_names, metadata = ml_model.load_model_artifacts()

# Get current features and prediction
current_features = st.session_state.current_features
original_prediction = st.session_state.current_prediction

st.info(f"""
📊 **Current Baseline Prediction:** {original_prediction:.2f} / 20
Explore how changing different factors would affect this prediction.
""")

st.markdown("---")

# What-if form
selected_feature, new_value = forms.create_what_if_form(current_features)

st.markdown("---")

if st.button("🔮 Simulate Impact"):
    with st.spinner("Simulating scenario..."):
        # Simulate what-if
        result = ml_model.simulate_what_if(
            current_features,
            model,
            scaler,
            encoder_dict,
            feature_names,
            selected_feature,
            new_value
        )
        
        # Display results
        cards.display_comparison_card(
            result['original_prediction'],
            result['new_prediction'],
            selected_feature
        )
        
        st.markdown("---")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_comparison = charts.create_comparison_chart(
                result['original_prediction'],
                result['new_prediction'],
                selected_feature
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            # Display impact summary
            st.markdown("### 📈 Impact Summary")
            
            impact = result['impact']
            pct_change = result['percentage_change']
            
            st.metric("Grade Change", f"{impact:+.2f} points")
            st.metric("Percentage Change", f"{pct_change:+.1f}%")
            
            if impact > 0:
                st.success(f"✅ This change would IMPROVE the grade")
            elif impact < 0:
                st.error(f"⚠️ This change would DECREASE the grade")
            else:
                st.info("ℹ️ No significant impact")
        
        st.markdown("---")
        
        # Additional insights
        st.markdown("### 💡 Insights")
        
        if selected_feature == 'studytime':
            if impact > 0:
                st.write("""
                📚 Increasing study time is one of the most effective ways to improve grades. 
                Consider establishing a consistent study routine and finding a quiet study environment.
                """)
        elif selected_feature == 'absences':
            if impact < 0:
                st.write("""
                🏫 Reducing absences is crucial for academic success. Regular attendance ensures 
                students don't miss important lessons and stay engaged with the material.
                """)
        elif selected_feature in ['Dalc', 'Walc']:
            if impact < 0:
                st.write("""
                🚫 Alcohol consumption negatively impacts academic performance. Reducing consumption 
                can improve focus, memory, and overall academic outcomes.
                """)
        elif selected_feature == 'health':
            if impact > 0:
                st.write("""
                💪 Better health leads to better academic performance. Encourage regular exercise, 
                proper nutrition, and adequate sleep.
                """)

st.markdown("---")

# Multiple scenario comparison
st.markdown("### 📊 Multiple Scenario Comparison")

with st.expander("🔬 Compare Multiple Changes"):
    st.write("Compare the impact of changing multiple features simultaneously")
    
    col1, col2 = st.columns(2)
    
    scenarios = {}
    
    with col1:
        st.markdown("#### Scenario 1: Improved Study Habits")
        scenarios['study_habits'] = {
            'studytime': min(current_features.get('studytime', 2) + 1, 4),
            'absences': max(current_features.get('absences', 0) - 5, 0)
        }
    
    with col2:
        st.markdown("#### Scenario 2: Better Health & Lifestyle")
        scenarios['health'] = {
            'health': min(current_features.get('health', 3) + 1, 5),
            'Dalc': max(current_features.get('Dalc', 1) - 1, 1),
            'Walc': max(current_features.get('Walc', 1) - 1, 1)
        }
    
    if st.button("Compare Scenarios"):
        results_data = {'Scenario': ['Original'], 'Prediction': [original_prediction]}
        
        for scenario_name, changes in scenarios.items():
            modified_features = current_features.copy()
            modified_features.update(changes)
            
            pred = ml_model.predict_single(
                modified_features, model, scaler, encoder_dict, feature_names
            )
            
            results_data['Scenario'].append(scenario_name.replace('_', ' ').title())
            results_data['Prediction'].append(pred)
        
        import pandas as pd
        results_df = pd.DataFrame(results_data)
        
        st.dataframe(results_df)
        
        # Visualization
        import plotly.express as px
        fig = px.bar(
            results_df,
            x='Scenario',
            y='Prediction',
            title='Scenario Comparison',
            color='Prediction',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis=dict(range=[0, 20]))
        st.plotly_chart(fig, use_container_width=True)