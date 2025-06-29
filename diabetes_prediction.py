import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

diab = pd.read_csv("diabetes.csv")

X = diab.drop(columns='Outcome', axis=1)
Y = diab['Outcome']

scaler = StandardScaler()
scaler.fit(X)
Standardized_data = scaler.transform(X)
X = Standardized_data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f"Training Data Accuracy: {training_data_accuracy:.2%}")

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Test Data Accuracy: {test_data_accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))

input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

standard_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(standard_data)

if prediction[0] == 0:
    print("NON DIABETIC :)")
else:
    print("DIABETIC :(")

print(f"Prediction: {prediction[0]}") 