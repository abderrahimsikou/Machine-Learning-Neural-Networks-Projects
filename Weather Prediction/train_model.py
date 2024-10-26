import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score 
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
import joblib

# Read data
data = pd.read_csv('train_dataset.csv')

# Drop unnecessary column
data = data.drop('date',axis=1)

# Convert from textual to numerique
convert = LabelEncoder()
data['weather'] = convert.fit_transform(data['weather'])

# Splitt data into features and target
x = data.drop(['weather'], axis=1)
y = data['weather']

# Splitt data
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2,random_state=30)

# Scaler data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build modle
model = GradientBoostingClassifier(n_estimators=100 , random_state=30 , learning_rate=0.04)
model.fit(x_train,y_train)

#Show Results of the model {accuracy}
prediction = model.predict(x_test)
prediction2 = model.predict(x_train)

accuarcy = accuracy_score(prediction,y_test)
print('accuracy_score',accuarcy * 100, '%')
accuarcy2 = accuracy_score(prediction2,y_train)
print('accuracy_score',accuarcy2 * 100, '%')


val = cross_val_score(model, x,y,cv=5)
print('cross validation scores:', val)
print('average cross validation scores:', val.mean())

cm = confusion_matrix(prediction,y_test)
print('cm:\n', cm)

classification = classification_report(prediction, y_test)
print('classification_report:\n' , classification)

# Save model and standardscaler
joblib.dump(model , 'Model.pkl')
joblib.dump(scaler , 'scaler.pkl')