"""
Analytics Dashboard Page
"""

import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils
from components import charts, cards

st.set_page_config(**config.PAGE_CONFIG)

st.title("📊 Analytics Dashboard")

# Load data
if not os.path.exists(config.DATA_PATH):
    st.error("❌ Dataset not found!")
    st.stop()

df = utils.load_data()

if df is None:
    st.error("❌ Error loading dataset!")
    st.stop()

st.success(f"✅ Dataset loaded: {len(df)} students")

# Overview statistics
st.markdown("### 📈 Overview Statistics")
cards.display_statistics_card(df)

st.markdown("---")

# Grade distribution
st.markdown("### 📊 Grade Distribution")
col1, col2 = st.columns(2)

with col1:
    fig_dist = charts.create_grade_distribution(df)
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    # Add risk levels to dataframe
    df['risk_level'] = df['G3'].apply(utils.get_risk_level)
    fig_risk = charts.create_risk_distribution_pie(df)
    st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")

# Correlation analysis
st.markdown("### 🔗 Feature Correlations")

with st.expander("View Correlation Heatmap"):
    fig_corr = charts.create_correlation_heatmap(df)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.info("""
    💡 **Interpretation:** 
    - Strong positive correlations (blue) indicate features that increase together
    - Strong negative correlations (red) indicate features that move in opposite directions
    - G1 and G2 (previous grades) typically show the strongest correlation with G3 (final grade)
    """)

st.markdown("---")

# Key metrics analysis
st.markdown("### 🎯 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Study Time Impact")
    fig_study = charts.create_box_plot(df, 'studytime')
    st.plotly_chart(fig_study, use_container_width=True)

with col2:
    st.markdown("#### Failures Impact")
    fig_failures = charts.create_box_plot(df, 'failures')
    st.plotly_chart(fig_failures, use_container_width=True)

with col3:
    st.markdown("#### Health Impact")
    fig_health = charts.create_box_plot(df, 'health')
    st.plotly_chart(fig_health, use_container_width=True)

st.markdown("---")

# Comparative analysis
st.markdown("### 🔍 Comparative Analysis")

analysis_type = st.selectbox(
    "Select Analysis Type",
    [
        "Gender Comparison",
        "Urban vs Rural",
        "Internet Access Impact",
        "Higher Education Aspiration",
        "Family Support Impact"
    ]
)

if analysis_type == "Gender Comparison":
    col1, col2 = st.columns(2)
    
    with col1:
        male_avg = df[df['sex'] == 'M']['G3'].mean()
        female_avg = df[df['sex'] == 'F']['G3'].mean()
        
        st.metric("Male Average Grade", f"{male_avg:.2f}")
        st.metric("Female Average Grade", f"{female_avg:.2f}")
        st.metric("Difference", f"{abs(male_avg - female_avg):.2f}")
    
    with col2:
        gender_data = df.groupby('sex')['G3'].mean().reset_index()
        import plotly.express as px
        fig = px.bar(
            gender_data,
            x='sex',
            y='G3',
            title='Average Grade by Gender',
            labels={'sex': 'Gender', 'G3': 'Average Grade'},
            color='G3',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Urban vs Rural":
    col1, col2 = st.columns(2)
    
    with col1:
        urban_avg = df[df['address'] == 'U']['G3'].mean()
        rural_avg = df[df['address'] == 'R']['G3'].mean()
        
        st.metric("Urban Average Grade", f"{urban_avg:.2f}")
        st.metric("Rural Average Grade", f"{rural_avg:.2f}")
        st.metric("Difference", f"{abs(urban_avg - rural_avg):.2f}")
    
    with col2:
        address_data = df.groupby('address')['G3'].mean().reset_index()
        address_data['address'] = address_data['address'].map({'U': 'Urban', 'R': 'Rural'})
        import plotly.express as px
        fig = px.bar(
            address_data,
            x='address',
            y='G3',
            title='Average Grade by Address Type',
            labels={'address': 'Address Type', 'G3': 'Average Grade'},
            color='G3',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Internet Access Impact":
    col1, col2 = st.columns(2)
    
    with col1:
        with_internet = df[df['internet'] == 'yes']['G3'].mean()
        without_internet = df[df['internet'] == 'no']['G3'].mean()
        
        st.metric("With Internet", f"{with_internet:.2f}")
        st.metric("Without Internet", f"{without_internet:.2f}")
        st.metric("Difference", f"{abs(with_internet - without_internet):.2f}")
        
        if with_internet > without_internet:
            st.success("✅ Internet access shows positive correlation with grades")
    
    with col2:
        internet_data = df.groupby('internet')['G3'].mean().reset_index()
        internet_data['internet'] = internet_data['internet'].map({'yes': 'Yes', 'no': 'No'})
        import plotly.express as px
        fig = px.bar(
            internet_data,
            x='internet',
            y='G3',
            title='Impact of Internet Access on Grades',
            labels={'internet': 'Internet Access', 'G3': 'Average Grade'},
            color='G3',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Higher Education Aspiration":
    col1, col2 = st.columns(2)
    
    with col1:
        wants_higher = df[df['higher'] == 'yes']['G3'].mean()
        not_wants_higher = df[df['higher'] == 'no']['G3'].mean()
        
        st.metric("Wants Higher Education", f"{wants_higher:.2f}")
        st.metric("Doesn't Want Higher Education", f"{not_wants_higher:.2f}")
        st.metric("Difference", f"{abs(wants_higher - not_wants_higher):.2f}")
        
        st.info("💡 Students with higher education goals typically perform better")
    
    with col2:
        higher_data = df.groupby('higher')['G3'].mean().reset_index()
        higher_data['higher'] = higher_data['higher'].map({'yes': 'Yes', 'no': 'No'})
        import plotly.express as px
        fig = px.bar(
            higher_data,
            x='higher',
            y='G3',
            title='Impact of Higher Education Goals',
            labels={'higher': 'Wants Higher Education', 'G3': 'Average Grade'},
            color='G3',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Family Support Impact":
    support_combinations = df.groupby(['schoolsup', 'famsup'])['G3'].mean().reset_index()
    support_combinations['support_type'] = (
        support_combinations['schoolsup'].map({'yes': 'School ', 'no': 'No School '}) +
        support_combinations['famsup'].map({'yes': '+ Family', 'no': '+ No Family'})
    )
    
    import plotly.express as px
    fig = px.bar(
        support_combinations,
        x='support_type',
        y='G3',
        title='Impact of Support Systems on Grades',
        labels={'support_type': 'Support Type', 'G3': 'Average Grade'},
        color='G3',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Additional insights
st.markdown("### 💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Top Performing Students Characteristics")
    top_performers = df[df['G3'] >= 16]
    
    if len(top_performers) > 0:
        st.write(f"**Count:** {len(top_performers)} students")
        st.write(f"**Average Study Time:** {top_performers['studytime'].mean():.1f}")
        st.write(f"**Average Absences:** {top_performers['absences'].mean():.1f}")
        st.write(f"**Want Higher Education:** {(top_performers['higher'] == 'yes').sum() / len(top_performers) * 100:.1f}%")
        st.write(f"**Have Internet:** {(top_performers['internet'] == 'yes').sum() / len(top_performers) * 100:.1f}%")

with col2:
    st.markdown("#### At-Risk Students Characteristics")
    at_risk = df[df['G3'] < 10]
    
    if len(at_risk) > 0:
        st.write(f"**Count:** {len(at_risk)} students")
        st.write(f"**Average Study Time:** {at_risk['studytime'].mean():.1f}")
        st.write(f"**Average Absences:** {at_risk['absences'].mean():.1f}")
        st.write(f"**Average Failures:** {at_risk['failures'].mean():.1f}")
        st.write(f"**Want Higher Education:** {(at_risk['higher'] == 'yes').sum() / len(at_risk) * 100:.1f}%")