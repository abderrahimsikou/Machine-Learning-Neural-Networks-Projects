import pandas as pd
import joblib

# Upload dataset & model & StandardScaler
data = pd.read_csv('testing_dataset.csv')
model = joblib.load('model.pkl')
scaler = joblib.load('standardScaler.pkl')

# Data information
print(data.head())
print(data.isnull().sum())
print(data.dtypes)
print(data.duplicated().sum())

data = data.drop('name',axis=1)

# Scaler data
scaled_data = scaler.transform(data)

# Predict
prediction = model.predict(scaled_data)
print('Prediction for new data:',prediction)