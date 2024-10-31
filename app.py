import pandas as pd
import numpy as np
import streamlit as st
import pickle 

# Load data and model
@st.cache_data()
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('le.pkl', 'rb') as file:
        le = pickle.load(file)
    return model, scaler, le

# Title 
st.title("Hospital Excess Readmission Prediction")
st.write("""
### Predict if a hospital will have a 'High' or 'Low' Excess Readmission Ratio
This tool helps hospitals assess their readmission rates by entering specific data points below.
""")

# Measure Names
measure_name_options = [
    "Heart_Failure", 
    "Hip_Knee_Surgery", 
    "Pneumonia", 
    "Chronic_Obstructive_Pulmonary", 
    "Acute_Myocardial_Infarction"
]

# hospital input features
def hospital_input_features():
    st.write("#### Step 1: Select Measure Name")
    
    measure_name = st.selectbox("Measure Name", options=measure_name_options, help="Select the specific measure for the hospital's readmission assessment.")
    
    st.write("#### Step 2: Enter Discharge and Readmission Rates")
    num_discharges = st.number_input("Number of Discharges", min_value=0, value=500, step=1, help="Total number of hospital discharges.")
    predicted_readmission_rate = st.number_input("Predicted Readmission Rate", min_value=0.0, value=25.5, step=0.01, format='%.2f', help="Predicted rate of patient readmissions.")
    expected_readmission_rate = st.number_input("Expected Readmission Rate", min_value=0.0, value=25.4, step=0.01, format='%.2f', help="Expected rate of patient readmissions based on similar hospitals.")
    num_readmissions = st.number_input("Number of Readmissions", min_value=0, value=100, step=1, help="Total number of patient readmissions.")

    return {
        'Measure Name': measure_name,
        'Number of Discharges': num_discharges,
        'Predicted Readmission Rate': predicted_readmission_rate,
        'Expected Readmission Rate': expected_readmission_rate,
        'Number of Readmissions': num_readmissions,
    }

# Load model and scalers
model, scaler, le = load_model()
hospital_input = hospital_input_features()

# Prediction button
st.write("#### Step 3: Generate Prediction")
if st.button('Predict Readmissions'):
    try:
        # Encode and scale the input data
        encoded_measure_name = le.transform([hospital_input['Measure Name']])[0]
        input_data = [
            encoded_measure_name,
            hospital_input['Number of Discharges'],
            hospital_input['Predicted Readmission Rate'],
            hospital_input['Expected Readmission Rate'],
            hospital_input['Number of Readmissions']
        ]
        scaled_data = scaler.transform([input_data])

        # Predict the output
        prediction = model.predict(scaled_data)[0]

      
        if prediction == 'High':
            st.warning("Prediction: High Excess Readmission Rate!")
            st.write("Recommendation: Consider further investigation to reduce readmissions.")
        else:
            st.success("Prediction: Low Excess Readmission Rate!")
            st.write("Recommendation: This hospital is performing well according to current data.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Fill in all fields and click 'Predict Readmissions' to begin.")

with st.sidebar:
    st.header("About This App")
    st.write("""
        This application uses historical hospital data to estimate if a hospital's readmission rate is high or low. 
        Adjust the input data on the Input Data tab, and get results with the Prediction.
    """)
    st.write("### Want to learn more?")
    st.markdown("[Medicare Readmission Policy](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/HRRP/Hospital-Readmission-Reduction-Program)")
    st.write("The prediction provides guidance only and should not replace comprehensive assessments.")

# styling
st.markdown("""
    <style>
        .css-1aumxhk {background-color: #f9f9f9; padding: 10px; border-radius: 5px;}
        .stButton>button {background-color: #4CAF50; color: white; font-size: 16px;}
        .stTabs>div>div>button {font-size: 16px; color: #4CAF50;}
    </style>
""", unsafe_allow_html=True)
