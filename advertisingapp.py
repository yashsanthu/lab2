# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:28:38 2024

@author: GK986HL
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
 
# Load the pre-trained Linear Regression model
with open("advLinearRegression.pkl", "rb") as f:
    model = pickle.load(f)
 
# Create the Streamlit app
st.title("Sales Prediction App")
 
# Sidebar input
tv_budget = st.sidebar.slider("TV Budget (in thousands of dollars)", 0, 100, 50)
radio_budget = st.sidebar.slider("Radio Budget (in thousands of dollars)", 0, 100, 30)
newspaper_budget = st.sidebar.slider("Newspaper Budget (in thousands of dollars)", 0, 100, 20)
 
# Prepare input data
input_data = pd.DataFrame({
    "TV": [tv_budget],
    "Radio": [radio_budget],
    "Newspaper": [newspaper_budget]
})
 
# Make predictions
sales_prediction = model.predict(input_data)
 
# Display prediction
st.write(f"Predicted Sales: {sales_prediction[0]:.2f} thousand units")