import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
import joblib

#Read data
data = pd.read_csv('training_dataset.csv')

print(data.columns)             # to check the columns
print(data.isnull().sum())      # to check the missing values
print(data.dtypes)              # to check the textual columns
print(data.duplicated().sum())  # to check how the duplicated values

numerique = LabelEncoder()
data['gender'] = numerique.fit_transform(data['gender'])
data['smoking_history'] = numerique.fit_transform(data['smoking_history'])

#Prepares Data by splitting into features [x] and target [y]
x = data.drop('diabetes', axis=1)
y = data['diabetes']

#Split data
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = GradientBoostingClassifier()
model.fit(x_train,y_train)

test_prediction= model.predict(x_test)
train_prediction= model.predict(x_train)

test_accuracy = accuracy_score(test_prediction,y_test)
train_accuracy = accuracy_score(train_prediction,y_train)


val = cross_val_score(model, x,y,cv=5)

cm = confusion_matrix(test_prediction,y_test)

classification_report = classification_report(test_prediction, y_test)

print('test  accuracy:', test_accuracy * 100, '%')
print('train accuracy:', train_accuracy * 100, '%')

print('cross validation scores:', val)
print('average cross validation scores:', val.mean())

print('cm:\n', cm)

print('classification_report:\n' , classification_report)

#Save model and StandardScaler
#joblib.dump(model, 'Model.pkl')
#joblib.dump(scaler, 'StandardScaler.pkl')