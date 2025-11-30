"""
Form components for Student Performance Prediction System
"""

import streamlit as st
import config

def create_student_input_form():
    """Create comprehensive student information input form"""
    
    st.subheader("📝 Student Information")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    # Demographics
    with col1:
        st.markdown("#### Demographics")
        school = st.selectbox(
            "School",
            options=['GP', 'MS'],
            help=config.FEATURE_DESCRIPTIONS['school'],
            key='school_input'
        )
        
        sex = st.selectbox(
            "Gender",
            options=['M', 'F'],
            help=config.FEATURE_DESCRIPTIONS['sex'],
            key='sex_input'
        )
        
        age = st.number_input(
            "Age",
            min_value=15,
            max_value=22,
            value=17,
            help=config.FEATURE_DESCRIPTIONS['age'],
            key='age_input'
        )
        
        address = st.selectbox(
            "Address Type",
            options=['U', 'R'],
            format_func=lambda x: 'Urban' if x == 'U' else 'Rural',
            help=config.FEATURE_DESCRIPTIONS['address'],
            key='address_input'
        )
        
        famsize = st.selectbox(
            "Family Size",
            options=['LE3', 'GT3'],
            format_func=lambda x: '≤ 3 members' if x == 'LE3' else '> 3 members',
            help=config.FEATURE_DESCRIPTIONS['famsize'],
            key='famsize_input'
        )
        
        Pstatus = st.selectbox(
            "Parents Status",
            options=['T', 'A'],
            format_func=lambda x: 'Living together' if x == 'T' else 'Apart',
            help=config.FEATURE_DESCRIPTIONS['Pstatus'],
            key='pstatus_input'
        )
    
    # Family background
    with col2:
        st.markdown("#### Family Background")
        Medu = st.slider(
            "Mother's Education",
            min_value=0,
            max_value=4,
            value=2,
            help=config.FEATURE_DESCRIPTIONS['Medu'],
            key='medu_input'
        )
        
        Fedu = st.slider(
            "Father's Education",
            min_value=0,
            max_value=4,
            value=2,
            help=config.FEATURE_DESCRIPTIONS['Fedu'],
            key='fedu_input'
        )
        
        Mjob = st.selectbox(
            "Mother's Job",
            options=config.JOB_CATEGORIES,
            help=config.FEATURE_DESCRIPTIONS['Mjob'],
            key='mjob_input'
        )
        
        Fjob = st.selectbox(
            "Father's Job",
            options=config.JOB_CATEGORIES,
            help=config.FEATURE_DESCRIPTIONS['Fjob'],
            key='fjob_input'
        )
        
        guardian = st.selectbox(
            "Guardian",
            options=config.GUARDIAN_CATEGORIES,
            help=config.FEATURE_DESCRIPTIONS['guardian'],
            key='guardian_input'
        )
        
        traveltime = st.slider(
            "Travel Time to School",
            min_value=1,
            max_value=4,
            value=2,
            help=config.FEATURE_DESCRIPTIONS['traveltime'],
            key='traveltime_input'
        )
    
    # Academic information
    with col3:
        st.markdown("#### Academic History")
        studytime = st.slider(
            "Weekly Study Time",
            min_value=1,
            max_value=4,
            value=2,
            help=config.FEATURE_DESCRIPTIONS['studytime'],
            key='studytime_input'
        )
        
        failures = st.number_input(
            "Past Class Failures",
            min_value=0,
            max_value=4,
            value=0,
            help=config.FEATURE_DESCRIPTIONS['failures'],
            key='failures_input'
        )
        
        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=93,
            value=0,
            help=config.FEATURE_DESCRIPTIONS['absences'],
            key='absences_input'
        )
        
        G1 = st.number_input(
            "First Period Grade (G1)",
            min_value=0,
            max_value=20,
            value=10,
            help=config.FEATURE_DESCRIPTIONS['G1'],
            key='g1_input'
        )
        
        G2 = st.number_input(
            "Second Period Grade (G2)",
            min_value=0,
            max_value=20,
            value=10,
            help=config.FEATURE_DESCRIPTIONS['G2'],
            key='g2_input'
        )
        
        reason = st.selectbox(
            "Reason for Choosing School",
            options=config.REASON_CATEGORIES,
            help=config.FEATURE_DESCRIPTIONS['reason'],
            key='reason_input'
        )
    
    # Support and activities
    st.markdown("#### 🤝 Support & Activities")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        schoolsup = st.selectbox(
            "School Support",
            options=['yes', 'no'],
            help=config.FEATURE_DESCRIPTIONS['schoolsup'],
            key='schoolsup_input'
        )
        
        famsup = st.selectbox(
            "Family Support",
            options=['yes', 'no'],
            help=config.FEATURE_DESCRIPTIONS['famsup'],
            key='famsup_input'
        )
        
        paid = st.selectbox(
            "Extra Paid Classes",
            options=['yes', 'no'],
            help=config.FEATURE_DESCRIPTIONS['paid'],
            key='paid_input'
        )
    
    with col5:
        activities = st.selectbox(
            "Extra-curricular Activities",
            options=['yes', 'no'],
            help=config.FEATURE_DESCRIPTIONS['activities'],
            key='activities_input'
        )
        
        nursery = st.selectbox(
            "Attended Nursery",
            options=['yes', 'no'],
            help=config.FEATURE_DESCRIPTIONS['nursery'],
            key='nursery_input'
        )
        
        higher = st.selectbox(
            "Wants Higher Education",
            options=['yes', 'no'],
            help=config.FEATURE_DESCRIPTIONS['higher'],
            key='higher_input'
        )
    
    with col6:
        internet = st.selectbox(
            "Internet at Home",
            options=['yes', 'no'],
            help=config.FEATURE_DESCRIPTIONS['internet'],
            key='internet_input'
        )
        
        romantic = st.selectbox(
            "In Relationship",
            options=['yes', 'no'],
            help=config.FEATURE_DESCRIPTIONS['romantic'],
            key='romantic_input'
        )
    
    # Lifestyle factors
    st.markdown("#### 🎯 Lifestyle & Health")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        famrel = st.slider(
            "Family Relationships Quality",
            min_value=1,
            max_value=5,
            value=4,
            help=config.FEATURE_DESCRIPTIONS['famrel'],
            key='famrel_input'
        )
        
        freetime = st.slider(
            "Free Time After School",
            min_value=1,
            max_value=5,
            value=3,
            help=config.FEATURE_DESCRIPTIONS['freetime'],
            key='freetime_input'
        )
    
    with col8:
        goout = st.slider(
            "Going Out with Friends",
            min_value=1,
            max_value=5,
            value=3,
            help=config.FEATURE_DESCRIPTIONS['goout'],
            key='goout_input'
        )
        
        health = st.slider(
            "Current Health Status",
            min_value=1,
            max_value=5,
            value=4,
            help=config.FEATURE_DESCRIPTIONS['health'],
            key='health_input'
        )
    
    with col9:
        Dalc = st.slider(
            "Workday Alcohol Consumption",
            min_value=1,
            max_value=5,
            value=1,
            help=config.FEATURE_DESCRIPTIONS['Dalc'],
            key='dalc_input'
        )
        
        Walc = st.slider(
            "Weekend Alcohol Consumption",
            min_value=1,
            max_value=5,
            value=1,
            help=config.FEATURE_DESCRIPTIONS['Walc'],
            key='walc_input'
        )
    
    # Collect all form data
    form_data = {
        'school': school,
        'sex': sex,
        'age': age,
        'address': address,
        'famsize': famsize,
        'Pstatus': Pstatus,
        'Medu': Medu,
        'Fedu': Fedu,
        'Mjob': Mjob,
        'Fjob': Fjob,
        'reason': reason,
        'guardian': guardian,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'activities': activities,
        'nursery': nursery,
        'higher': higher,
        'internet': internet,
        'romantic': romantic,
        'famrel': famrel,
        'freetime': freetime,
        'goout': goout,
        'Dalc': Dalc,
        'Walc': Walc,
        'health': health,
        'absences': absences,
        'G1': G1,
        'G2': G2
    }
    
    return form_data

def create_quick_prediction_form():
    """Create simplified quick prediction form with essential features"""
    
    st.subheader("🚀 Quick Prediction")
    st.info("Fill in key information for a fast prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 15, 22, 17, key='quick_age')
        sex = st.selectbox("Gender", ['M', 'F'], key='quick_sex')
        studytime = st.slider("Study Time (1-4)", 1, 4, 2, key='quick_study')
        failures = st.number_input("Past Failures", 0, 4, 0, key='quick_failures')
        absences = st.number_input("Absences", 0, 93, 0, key='quick_absences')
    
    with col2:
        G1 = st.number_input("First Period Grade", 0, 20, 10, key='quick_g1')
        G2 = st.number_input("Second Period Grade", 0, 20, 10, key='quick_g2')
        higher = st.selectbox("Wants Higher Education", ['yes', 'no'], key='quick_higher')
        internet = st.selectbox("Internet at Home", ['yes', 'no'], key='quick_internet')
        health = st.slider("Health Status (1-5)", 1, 5, 4, key='quick_health')
    
    # Fill in default values for other required features
    form_data = {
        'school': 'GP',
        'sex': sex,
        'age': age,
        'address': 'U',
        'famsize': 'GT3',
        'Pstatus': 'T',
        'Medu': 2,
        'Fedu': 2,
        'Mjob': 'other',
        'Fjob': 'other',
        'reason': 'course',
        'guardian': 'mother',
        'traveltime': 2,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': 'no',
        'famsup': 'yes',
        'paid': 'no',
        'activities': 'no',
        'nursery': 'yes',
        'higher': higher,
        'internet': internet,
        'romantic': 'no',
        'famrel': 4,
        'freetime': 3,
        'goout': 3,
        'Dalc': 1,
        'Walc': 1,
        'health': health,
        'absences': absences,
        'G1': G1,
        'G2': G2
    }
    
    return form_data

def create_what_if_form(current_features):
    """Create form for what-if analysis"""
    
    st.subheader("🔮 What-If Analysis")
    st.info("Select a feature to modify and see how it affects the prediction")
    
    # Select feature to modify
    modifiable_features = {
        'studytime': 'Study Time',
        'absences': 'Absences',
        'failures': 'Past Failures',
        'health': 'Health Status',
        'freetime': 'Free Time',
        'goout': 'Going Out',
        'Dalc': 'Workday Alcohol',
        'Walc': 'Weekend Alcohol',
        'famrel': 'Family Relationships'
    }
    
    selected_feature = st.selectbox(
        "Feature to Modify",
        options=list(modifiable_features.keys()),
        format_func=lambda x: modifiable_features[x]
    )
    
    current_value = current_features.get(selected_feature, 0)
    
    st.write(f"**Current Value:** {current_value}")
    
    # Create appropriate input based on feature type
    if selected_feature == 'absences':
        new_value = st.number_input(
            "New Value",
            min_value=0,
            max_value=93,
            value=int(current_value)
        )
    elif selected_feature == 'failures':
        new_value = st.number_input(
            "New Value",
            min_value=0,
            max_value=4,
            value=int(current_value)
        )
    else:
        new_value = st.slider(
            "New Value",
            min_value=1,
            max_value=5,
            value=int(current_value)
        )
    
    return selected_feature, new_value