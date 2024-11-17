import pandas as pd
import joblib

# Read data
data = pd.read_csv('testing_dataset.csv')

# Upload Model
model = joblib.load('model.pkl')

#Predict
prediction = model.predict(data)
print('Prediction for new data:',prediction)