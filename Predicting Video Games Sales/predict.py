import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

#Upload Model StandardScaler
model = joblib.load('Model.pkl')
scaler = joblib.load('scaler.pkl')

#Read Data
data = pd.read_csv('test_dataset.csv')

#drop [Delete] some columns
data = data.drop(['rank','name','year'],axis=1)

#conversion textual rows into numeric
encoder = LabelEncoder()
data['platform'] = encoder.fit_transform(data['platform'].astype('str')) 
data['genre'] = encoder.fit_transform(data['genre'].astype('str'))
data['publisher'] = encoder.fit_transform(data['publisher'].astype('str'))

#scaler new data
scaled_data = scaler.transform(data)

#Predict
prediction = model.predict(scaled_data)
print('Prediction for new data:',prediction)