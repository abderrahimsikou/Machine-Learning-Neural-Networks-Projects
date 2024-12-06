import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

model = joblib.load('model.pkl')
data  = pd.read_csv('testing_dataset.csv')

# Data Information
print(data.head())                
print(data.isnull().sum())
print(data.dtypes)
print(data.duplicated().sum())

# Convert from textual into numerique
label_encoder = LabelEncoder()

data['Fuel_Type']    = label_encoder.fit_transform(data['Fuel_Type'])         
data['Seller_Type']  = label_encoder.fit_transform(data['Seller_Type'])
data['Transmission'] = label_encoder.fit_transform(data['Transmission'])

prediction = model.predict(data)

print('Result of new data:', prediction)