import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
import joblib

# Read data
data = pd.read_csv('training_dataset.csv')

# Data information
print(data.head())
print(data.isnull().sum())
print(data.dtypes)
print(data.duplicated().sum())

data = data.drop('name',axis=1)

# Data correlation
coor = data.corr()
print(coor['status'].sort_values(ascending=False))

# Split data
x = data.drop('status',axis=1)
y = data['status']

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2,random_state=42)

# Scaler data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Train model
model = XGBClassifier()
model.fit(x_train,y_train)

prediction = model.predict(x_test)

accuracy = accuracy_score(prediction,y_test)

val = cross_val_score(model, x,y,cv=5)

cm = confusion_matrix(prediction,y_test)

classification = classification_report(prediction, y_test)

print('accuracy:', accuracy * 100, '%')

print('cross validation scores:', val)
print('average cross validation scores:', val.mean() * 100,'%')

print('cm:\n', cm)

print('classification_report:\n' , classification)

# Download model & StandardScaler
#joblib.dump(model, 'model.pkl')
#joblib.dump(scaler, 'StandardScaler.pkl')