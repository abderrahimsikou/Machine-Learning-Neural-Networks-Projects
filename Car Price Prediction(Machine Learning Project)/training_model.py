import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

data = pd.read_csv('training_dataset.csv')

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

#Split data
x = data.drop(['Car_Name','Selling_Price'],axis=1)
y = data['Selling_Price']

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2,random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(x_train,y_train)

prediction    = model.predict(x_test)

mean_absolute = mean_absolute_error(prediction, y_test)

mean_squared  = mean_squared_error(prediction, y_test)

r2score       = r2_score(prediction, y_test)

print('mean_absolute_error:', mean_absolute * 100, '%')
print('mean_squared_error:' , mean_squared * 100, '%')
print('r2_score:'           , r2score * 100, '%')

# Save mode
joblib.dump(model,'model.pkl')