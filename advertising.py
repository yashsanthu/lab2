# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:13:13 2024

@author: GK986HL
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 
# Load the 'advertising' dataset
 
data = pd.read_csv(r'C:\Users\GK986HL\Downloads\advertising.csv')
 
# Prepare X (features) and Y (target)
X = data[["TV", "Radio", "Newspaper"]]
Y = data["Sales"]
 
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
 
# Initialize the Linear Regression model
model = LinearRegression()
 
# Fit the model to the training data
model.fit(X_train, Y_train)
 
# Make predictions on the test data
Y_prediction = model.predict(X_test)
 
# Calculate metrics
mse = mean_squared_error(Y_test, Y_prediction)
r2 = r2_score(Y_test, Y_prediction)
 
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
 
# Save the model weights to a file
with open("advLinearRegression.pkl", "wb") as f:
    pickle.dump(model, f)