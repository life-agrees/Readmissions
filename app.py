import pandas as pd
import numpy as np
import streamlit as st
import pickle 

# Load data

@st.cache_data()

def load_model():
  with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    with open('scaler.pkl','rb')as file:
      scaler = pickle.load(file)
      with open('le.pkl','rb')as file:
        le = pickle.load(file)
        return model, scaler, le

#title of our app

st.title("Hospital Excess Readmission Prediction")

st.write("""
### This app predicts whether a hospital will have a 'High' or 'Low' Excess Readmission Ratio
Enter the hospital details in the sidebar, and press 'Predict' to get the prediction.
""")

#creating a sidebar for inputing data

st.sidebar.subheader("Enter hospital data")

#inputting data
def hospital_input_features():
  
  measure_name = st.sidebar.text_input("Measure Name","",help="Enter the specific measure name used for the hospital's readmission assessment.")
  num_discharges = st.sidebar.number_input("Number of Discharges", min_value=0, value=500,step=1,help="Enter the total number of hospital discharges.")
  predicted_readmission_rate = st.sidebar.number_input("Predicted Readmission Rate", min_value=0.0, value=25.5,step = 0.01,format= '%.2f',help="Enter the predicted readmission rate.")
  expected_readmission_rate = st.sidebar.number_input("Expected Readmission Rate", min_value=0.0, value=25.4,step= 0.01,format= '%.2f',help="Enter the expected readmission rate.")
  num_readmissions = st.sidebar.number_input("Number of Readmissions", min_value=0, value=100,step=1,help="Enter the total number of readmissions.")

  return{
    'Measure Name': measure_name,
    'Number of Discharges': num_discharges,
    'Predicted Readmission Rate': predicted_readmission_rate,
    'Expected Readmission Rate': expected_readmission_rate,
    'Number of Readmissions': num_readmissions,
  }

#load model

model, scaler, le = load_model()

hospital_input = hospital_input_features()

if st.sidebar.button('Predict'):
  if hospital_input ['Measure Name']:
    try:
      # encode the measure name
      encoded_measure_name = le.transform([hospital_input['Measure Name']])[0]
      input_data = [
        encoded_measure_name,
        hospital_input['Number of Discharges'],
        hospital_input['Predicted Readmission Rate'],
        hospital_input['Expected Readmission Rate'],
        hospital_input['Number of Readmissions']
        ]
      #scale data
      scaled_data = scaler.transform([input_data])

      #predict the output
      prediction = model.predict(scaled_data)[0]

      if prediction[0] == 'High' :
        status = 'High!'
        st.warning("Excess Readmission Rate is high!")
        st.write("This hospital may require further investigation.")
        st.write("Consider consulting with a hospital management team or a hospital statistician.")

      else:
        status = 'Low!'
        st.success("Excess Readmission Rate is low!")
        st.write("This hospital may be performing well according to its current data.")
        st.write("Consider taking a closer look at its performance.")
    except Exception as e:
      st.error(f"An error occurred: {str(e)}")
else:
  st.warning("Please enter a Measure Name to make prediction.")

