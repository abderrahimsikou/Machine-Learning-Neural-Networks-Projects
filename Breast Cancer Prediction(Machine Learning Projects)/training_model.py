import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
import joblib

data = pd.read_csv('training_dataset.csv')

print(data.columns)             # to check the columns
print(data.isnull().sum())      # to check the missing values
print(data.dtypes)              # to check the textual columns
print(data.duplicated().sum())  # to check how the duplicated values

numerique = LabelEncoder()
data['diagnosis'] = numerique.fit_transform(data['diagnosis'])

coor = data.corr()
print(coor['diagnosis'].sort_values(ascending=False))

x = data.drop('diagnosis',axis=1)
y = data['diagnosis']

#Split data
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(x_train,y_train)

prediction= model.predict(x_test)

accuracy = accuracy_score(prediction,y_test)

val = cross_val_score(model, x,y,cv=5)

cm = confusion_matrix(prediction,y_test)

classification = classification_report(prediction, y_test)

print('accuracy:', accuracy * 100, '%')

print('cross validation scores:', val)
print('average cross validation scores:', val.mean())

print('cm:\n', cm)

print('classification_report:\n' , classification)

# Save model
joblib.dump(model,'model.pkl')