import pandas as pd
import joblib

#upload the model and scaler
model = joblib.load('heart disease model.pkl')
scaler = joblib.load('StandareScaler.pkl')

#Read New Data
new_data = pd.read_csv('testing_dataset.csv')

#scaler new data
scaled_data = scaler.transform(new_data)

#Predict
prediction = model.predict(scaled_data)
print('Prediction for new data:',prediction)