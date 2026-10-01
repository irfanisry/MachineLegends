# MODEL: XGBoost
# Objective: To create ML predictive model in .joblib file format.

# NOTE:
# 01. No scaling (higher MSE when scaled).
# 02. Date column: 'Description'
# 03. Target column: 'iBA_in_nBA'

# Importing the required libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pickle
import numpy as np

# Load the dataset
Data = pd.read_csv('CaseStudy1_5mins_2022_40+_Cleaning_02.csv')

# Combine and split the dataset into input features (X) and target variable (y)
Data.drop('Description', axis=1, inplace=True)
X = Data.drop('iBA_in_nBA', axis=1)
y = Data['iBA_in_nBA']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Train the XGBoost model on the training set
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_predict = model.predict(X_test)

# Evaluate performance of model
# R-squared, Test Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
r2 = r2_score(y_test, y_predict)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, y_predict)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

# Print results
print("Model: XGBoost")
print(f"R-squared: {r2:.2f}")
print(f"Train score: {train_score:.2f}")
print(f"Test score: {test_score:.2f}")
print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Train RMSE: {train_mse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print("")

# Perform feature importance analysis
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature Importance")
for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))
    #print("")

# Save the model
joblib.dump(model, 'xgboost_model_Joblib_PowerBI.joblib')
