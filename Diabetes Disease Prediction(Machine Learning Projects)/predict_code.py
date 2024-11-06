from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib

#Read data
data = pd.read_csv('testing_dataset.csv')

#upload model and standardscaler
model  = joblib.load('Model.pkl')            # Model
scaler = joblib.load('StandardScaler.pkl')   # StandardScaler

print(data.columns)                          # to check the columns
print(data.isnull().sum())                   # to check the missing values
print(data.dtypes)                           # to check the textual columns
print(data.duplicated().sum())               # to check how the duplicated values

#convert from textual into numerique
numerique = LabelEncoder()
data['gender'] = numerique.fit_transform(data['gender'])
data['smoking_history'] = numerique.fit_transform(data['smoking_history'])

#scaler data
scaled_data = scaler.transform(data)

#Predict
prediction = model.predict(scaled_data)
print('Prediction for new data:',prediction)
