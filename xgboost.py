import streamlit as st
import xgboost as xgb
import pandas as pd
import joblib

# Load model
model = xgb.Booster()
model.load_model('your_new_model.bin')
model.set_param({'predictor': 'cpu_predictor'})  # Ensure it's using CPU

# Load preprocessed data (if necessary)
def preprocess_input(data):
    # Add your preprocessing steps here
    return data

# Prediction function
def predict(input_data):
    data = preprocess_input(input_data)
    dmatrix = xgb.DMatrix(data)
    return model.predict(dmatrix)

# Streamlit UI
st.title('Model Prediction')
input_data = st.text_input('Enter input data')

if input_data:
    prediction = predict(input_data)
    st.write(f'Prediction: {prediction}')
