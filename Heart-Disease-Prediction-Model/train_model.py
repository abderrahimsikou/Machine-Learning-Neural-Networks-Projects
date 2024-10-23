import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
import joblib

# Read Dataset 
data = pd.read_csv('training_dataset.csv')

#Prepares Data by splitting into features [x] and target [y]
x = data.drop('target', axis=1)
y = data['target']

#Split data
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.3,random_state=42)

#Scaler the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Training the model
model = RandomForestClassifier()
model.fit(x_train,y_train)

#Show Results of the model {accuracy}
prediction = model.predict(x_test)

accuarcy = accuracy_score(prediction,y_test)
print('accuracy_score',accuarcy * 100, '%')

val = cross_val_score(model, x,y,cv=5)
print('cross validation scores:', val)
print('average cross validation scores:', val.mean())

cm = confusion_matrix(prediction,y_test)
print('cm:\n', cm)

classification_report = classification_report(prediction, y_test)
print('classification_report:\n' , classification_report)

#save the model and scaler
#joblib.dump(model,'heart disease model.pkl')
#joblib.dump(scaler,'StandareScaler.pkl')
