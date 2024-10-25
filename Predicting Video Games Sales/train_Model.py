import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
import joblib

#read the data
data = pd.read_csv('train_dataset.csv')

#drop [Delete] some columns
data = data.drop(['rank','name','year'],axis=1)

#filling missing values 
data['publisher'] = data['publisher'].fillna(data.mean)

#conversion textual rows into numeric
encoder = LabelEncoder()
data['platform'] = encoder.fit_transform(data['platform'].astype('str')) 
data['genre'] = encoder.fit_transform(data['genre'].astype('str'))
data['publisher'] = encoder.fit_transform(data['publisher'].astype('str'))

#Prepares Data by splitting into features [x] and target [y]
x = data.drop(['global_sales'],axis=1)
y = data['global_sales']

#split the data
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.3,random_state=42)

#scaler the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#build model
model = LinearRegression()
model.fit(x_train,y_train)

#show the result of the model
prediction = model.predict(x_test)

mae = mean_absolute_error(prediction,y_test)
mse = mean_squared_error(prediction,y_test)
r = r2_score(prediction,y_test)

print('mse:', mae * 100, '%')
print('mse:', mse * 100, '%')
print('r2_score:', {r})

#Save the Model and the Standard Scaler
joblib.dump(model , 'Model.pkl')
joblib.dump(scaler , 'scaler.pkl')