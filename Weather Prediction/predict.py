import pandas as pd
import joblib

#Upload Model StandardScaler
model = joblib.load('Model.pkl')
scaler = joblib.load('scaler.pkl')

#Read Data
data = pd.read_csv('test_dataset.csv')

# Drop unnecessary column
data = data.drop('date',axis=1)

#scaler new data
scaled_data = scaler.transform(data)

#Predict
prediction = model.predict(scaled_data)
print('Prediction for new data:',prediction)