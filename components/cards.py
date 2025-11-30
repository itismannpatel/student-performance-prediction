"""
UI Card components for Student Performance Prediction System
"""

import streamlit as st
import config
import utils

def display_prediction_card(prediction, features):
    """Display prediction result in a styled card"""
    
    grade_category = utils.get_grade_category(prediction)
    risk_level = utils.get_risk_level(prediction)
    risk_score = utils.calculate_risk_score(prediction)
    
    # Main prediction display
    st.markdown("### 🎯 Prediction Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Predicted Grade",
            value=f"{prediction:.2f} / 20",
            delta=None
        )
    
    with col2:
        # Color code based on category
        color_map = {
            'Excellent': '🟢',
            'Good': '🟡',
            'Average': '🟠',
            'Below Average': '🟠',
            'Poor': '🔴'
        }
        st.metric(
            label="Grade Category",
            value=f"{color_map.get(grade_category, '')} {grade_category}"
        )
    
    with col3:
        risk_color = {
            'Low Risk': '🟢',
            'Medium Risk': '🟡',
            'High Risk': '🔴'
        }
        st.metric(
            label="Risk Level",
            value=f"{risk_color.get(risk_level, '')} {risk_level}"
        )
    
    with col4:
        st.metric(
            label="Risk Score",
            value=f"{risk_score}/100",
            delta=None,
            help="Higher score = Higher risk"
        )
    
    # Progress bar for grade
    st.markdown("#### Grade Performance")
    progress_value = prediction / 20
    st.progress(progress_value)
    
    # Additional context
    if prediction >= 16:
        st.success("🌟 Excellent performance! Student is performing exceptionally well.")
    elif prediction >= 14:
        st.success("✅ Good performance! Student is on track for success.")
    elif prediction >= 12:
        st.info("ℹ️ Average performance. Room for improvement with focused effort.")
    elif prediction >= 10:
        st.warning("⚠️ Below average. Student needs support and intervention.")
    else:
        st.error("🚨 High risk of failure. Immediate intervention recommended!")

def display_recommendations_card(recommendations):
    """Display personalized recommendations"""
    
    st.markdown("### 💡 Personalized Recommendations")
    
    if not recommendations:
        st.info("No specific recommendations - student is performing well!")
        return
    
    # Sort by priority
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    sorted_recommendations = sorted(
        recommendations,
        key=lambda x: priority_order.get(x['priority'], 3)
    )
    
    for rec in sorted_recommendations:
        priority = rec['priority']
        
        # Color code by priority
        if priority == 'High':
            border_color = "#d62728"
            icon = "🔴"
        elif priority == 'Medium':
            border_color = "#ff7f0e"
            icon = "🟡"
        else:
            border_color = "#2ca02c"
            icon = "🟢"
        
        with st.container():
            st.markdown(f"""
            <div style="
                border-left: 5px solid {border_color};
                padding: 10px;
                margin: 10px 0;
                background-color: #f0f2f6;
                border-radius: 5px;
            ">
                <strong>{icon} {rec['category']} - {priority} Priority</strong><br>
                <span style="font-size: 14px;">{rec['suggestion']}</span><br>
                <span style="font-size: 12px; color: #666;"><em>Expected Impact: {rec['impact']}</em></span>
            </div>
            """, unsafe_allow_html=True)

def display_student_profile_card(features):
    """Display student profile summary"""
    
    st.markdown("### 👤 Student Profile Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Academic Metrics")
        st.write(f"**Study Time:** {features.get('studytime', 0)}/4")
        st.write(f"**Previous Grades:** G1={features.get('G1', 0)}, G2={features.get('G2', 0)}")
        st.write(f"**Past Failures:** {features.get('failures', 0)}")
        st.write(f"**Absences:** {features.get('absences', 0)}")
        
        st.markdown("#### 🏠 Family Support")
        st.write(f"**Mother's Education:** {features.get('Medu', 0)}/4")
        st.write(f"**Father's Education:** {features.get('Fedu', 0)}/4")
        st.write(f"**Family Support:** {'✅' if features.get('famsup') == 'yes' else '❌'}")
        st.write(f"**School Support:** {'✅' if features.get('schoolsup') == 'yes' else '❌'}")
    
    with col2:
        st.markdown("#### 🎯 Personal Factors")
        st.write(f"**Age:** {features.get('age', 0)} years")
        st.write(f"**Health Status:** {features.get('health', 0)}/5")
        st.write(f"**Family Relationships:** {features.get('famrel', 0)}/5")
        st.write(f"**Higher Education Goal:** {'✅' if features.get('higher') == 'yes' else '❌'}")
        
        st.markdown("#### 🌐 Resources")
        st.write(f"**Internet Access:** {'✅' if features.get('internet') == 'yes' else '❌'}")
        st.write(f"**Extra Classes:** {'✅' if features.get('paid') == 'yes' else '❌'}")
        st.write(f"**Activities:** {'✅' if features.get('activities') == 'yes' else '❌'}")

def display_model_info_card(metadata):
    """Display model information and performance"""
    
    if not metadata:
        st.warning("Model metadata not available. Please train the model first.")
        return
    
    st.markdown("### 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Details")
        st.write(f"**Algorithm:** {metadata.get('model_type', 'Unknown')}")
        st.write(f"**Training Samples:** {metadata.get('training_samples', 0)}")
        st.write(f"**Test Samples:** {metadata.get('test_samples', 0)}")
        st.write(f"**Features Used:** {metadata.get('feature_count', 0)}")
        
        if 'saved_date' in metadata:
            st.write(f"**Last Trained:** {metadata['saved_date']}")
    
    with col2:
        st.markdown("#### Performance Metrics")
        metrics = metadata.get('metrics', {})
        
        st.metric("Mean Absolute Error (MAE)", f"{metrics.get('MAE', 0):.3f}")
        st.metric("Root Mean Squared Error (RMSE)", f"{metrics.get('RMSE', 0):.3f}")
        st.metric("R² Score", f"{metrics.get('R2', 0):.3f}")
        st.metric("Accuracy (±1 point)", f"{metrics.get('Accuracy', 0):.1f}%")

def display_comparison_card(original_pred, new_pred, feature_changed):
    """Display comparison results for what-if analysis"""
    
    st.markdown("### 📊 Comparison Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Original Prediction",
            value=f"{original_pred:.2f}"
        )
    
    with col2:
        st.metric(
            label="New Prediction",
            value=f"{new_pred:.2f}",
            delta=f"{new_pred - original_pred:.2f}"
        )
    
    with col3:
        change_pct = ((new_pred - original_pred) / original_pred * 100) if original_pred != 0 else 0
        st.metric(
            label="Percentage Change",
            value=f"{abs(change_pct):.1f}%",
            delta="Increase" if new_pred > original_pred else "Decrease"
        )
    
    # Impact analysis
    impact = new_pred - original_pred
    
    if impact > 1:
        st.success(f"✅ Significant positive impact! Changing {feature_changed} could improve the grade by {impact:.2f} points.")
    elif impact > 0:
        st.info(f"ℹ️ Moderate positive impact. Changing {feature_changed} could improve the grade by {impact:.2f} points.")
    elif impact < -1:
        st.error(f"⚠️ Significant negative impact! This change could decrease the grade by {abs(impact):.2f} points.")
    elif impact < 0:
        st.warning(f"⚠️ Slight negative impact. This change could decrease the grade by {abs(impact):.2f} points.")
    else:
        st.info("No significant change in prediction.")

def display_statistics_card(df):
    """Display dataset statistics"""
    
    st.markdown("### 📈 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    
    with col2:
        avg_grade = df['G3'].mean()
        st.metric("Average Grade", f"{avg_grade:.2f}")
    
    with col3:
        high_performers = len(df[df['G3'] >= 14])
        st.metric("High Performers", f"{high_performers} ({high_performers/len(df)*100:.1f}%)")
    
    with col4:
        at_risk = len(df[df['G3'] < 10])
        st.metric("At-Risk Students", f"{at_risk} ({at_risk/len(df)*100:.1f}%)")
    
    # Additional stats
    st.markdown("#### Grade Distribution")
    col5, col6, col7 = st.columns(3)
    
    with col5:
        excellent = len(df[df['G3'] >= 16])
        st.write(f"**Excellent (≥16):** {excellent} students")
    
    with col6:
        good = len(df[(df['G3'] >= 14) & (df['G3'] < 16)])
        st.write(f"**Good (14-15):** {good} students")
    
    with col7:
        average = len(df[(df['G3'] >= 12) & (df['G3'] < 14)])
        st.write(f"**Average (12-13):** {average} students")

def display_feature_impact_card(feature_importance_df):
    """Display top impactful features"""
    
    st.markdown("### 🎯 Most Impactful Features")
    
    if feature_importance_df is None or feature_importance_df.empty:
        st.info("Feature importance not available for this model type.")
        return
    
    top_5 = feature_importance_df.head(5)
    
    for idx, row in top_5.iterrows():
        feature = row['feature']
        importance = row['importance']
        
        st.markdown(f"""
        <div style="
            padding: 8px;
            margin: 5px 0;
            background-color: #f0f2f6;
            border-radius: 5px;
        ">
            <strong>{feature.replace('_', ' ').title()}</strong>
            <div style="
                background-color: #1f77b4;
                width: {importance * 100}%;
                height: 20px;
                border-radius: 3px;
                margin-top: 5px;
            "></div>
            <span style="font-size: 12px; color: #666;">{importance:.3f}</span>
        </div>
        """, unsafe_allow_html=True)