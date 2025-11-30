"""
Feature Analysis Page
"""

import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils
import model as ml_model
from components import charts, cards

st.set_page_config(**config.PAGE_CONFIG)

st.title("📈 Feature Analysis")

# Check if model exists
model_exists = (
    os.path.exists(config.MODEL_PATH) and
    os.path.exists(config.SCALER_PATH) and
    os.path.exists(config.ENCODER_PATH) and
    os.path.exists(config.FEATURE_NAMES_PATH)
)

if not model_exists:
    st.warning("⚠️ Please train the model first to view feature analysis!")
    st.stop()

# Load model and metadata
model, scaler, encoder_dict, feature_names, metadata = ml_model.load_model_artifacts()

if model is None or metadata is None:
    st.error("❌ Error loading model artifacts!")
    st.stop()

# Display model info
cards.display_model_info_card(metadata)

st.markdown("---")

# Feature Importance
st.markdown("### 🎯 Feature Importance Analysis")

feature_importance = ml_model.get_feature_importance(model, feature_names)

if feature_importance is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_importance = charts.create_feature_importance_chart(feature_importance, top_n=15)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        cards.display_feature_impact_card(feature_importance)
    
    st.markdown("---")
    
    # Detailed feature importance table
    with st.expander("📋 View Full Feature Importance Table"):
        st.dataframe(
            feature_importance.style.background_gradient(subset=['importance'], cmap='Blues'),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Feature importance insights
    st.markdown("### 💡 Key Findings")
    
    top_3_features = feature_importance.head(3)
    
    st.write("#### Top 3 Most Important Features:")
    
    for idx, row in top_3_features.iterrows():
        feature = row['feature']
        importance = row['importance']
        
        # Get feature description if available
        description = config.FEATURE_DESCRIPTIONS.get(feature, "No description available")
        
        st.markdown(f"""
        **{idx + 1}. {feature.replace('_', ' ').title()}**
        - Importance Score: {importance:.3f}
        - Description: {description}
        """)
    
    st.markdown("---")
    
    # Interactive feature exploration
    st.markdown("### 🔍 Interactive Feature Exploration")
    
    # Load data for exploration
    if os.path.exists(config.DATA_PATH):
        df = utils.load_data()
        
        if df is not None:
            # Select feature to explore
            numeric_features = [f for f in config.NUMERIC_FEATURES if f in df.columns]
            
            selected_feature = st.selectbox(
                "Select a feature to explore:",
                options=numeric_features,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot
                fig_scatter = charts.create_scatter_plot(df, selected_feature)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Statistics
                st.markdown(f"#### {selected_feature.replace('_', ' ').title()} Statistics")
                
                st.write(f"**Mean:** {df[selected_feature].mean():.2f}")
                st.write(f"**Median:** {df[selected_feature].median():.2f}")
                st.write(f"**Std Dev:** {df[selected_feature].std():.2f}")
                st.write(f"**Min:** {df[selected_feature].min():.2f}")
                st.write(f"**Max:** {df[selected_feature].max():.2f}")
                
                # Correlation with target
                correlation = df[[selected_feature, 'G3']].corr().iloc[0, 1]
                st.write(f"**Correlation with G3:** {correlation:.3f}")
                
                if abs(correlation) > 0.5:
                    st.success("Strong correlation!")
                elif abs(correlation) > 0.3:
                    st.info("Moderate correlation")
                else:
                    st.warning("Weak correlation")

else:
    st.info("""
    ℹ️ Feature importance is not available for this model type.
    Feature importance analysis works best with tree-based models like Random Forest or Gradient Boosting.
    """)

st.markdown("---")

# Model comparison
if 'all_model_evaluations' in metadata:
    st.markdown("### 🏆 Model Performance Comparison")
    
    all_evals = metadata['all_model_evaluations']
    fig_comparison = charts.create_multi_model_comparison(all_evals)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed metrics table
    with st.expander("📊 View Detailed Metrics"):
        comparison_df = pd.DataFrame(all_evals).T
        st.dataframe(
            comparison_df.style.highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE'])
                              .highlight_max(axis=0, subset=['R2', 'Accuracy']),
            use_container_width=True
        )

st.markdown("---")

# Engineering feature analysis
if os.path.exists(config.DATA_PATH):
    st.markdown("### 🔧 Engineered Features Analysis")
    
    df = utils.load_data()
    df_engineered = utils.calculate_engineered_features(df)
    
    st.info("""
    These are additional features created from the original data to potentially improve 
    prediction accuracy. They combine multiple original features to capture complex relationships.
    """)
    
    engineered_features = [
        'grade_trend',
        'parent_edu_avg',
        'total_alcohol',
        'support_score',
        'risk_factors',
        'activity_level',
        'home_environment'
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Engineered Features")
        for feat in engineered_features:
            if feat in df_engineered.columns:
                st.write(f"- **{feat.replace('_', ' ').title()}**")
    
    with col2:
        selected_eng_feature = st.selectbox(
            "Select engineered feature to explore:",
            options=[f for f in engineered_features if f in df_engineered.columns],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_eng_feature in df_engineered.columns:
            correlation = df_engineered[[selected_eng_feature, 'G3']].corr().iloc[0, 1]
            st.metric(
                "Correlation with Final Grade",
                f"{correlation:.3f}",
                delta=None
            )
    
    # Visualize selected engineered feature
    if selected_eng_feature in df_engineered.columns:
        fig = charts.create_scatter_plot(df_engineered, selected_eng_feature)
        st.plotly_chart(fig, use_container_width=True)