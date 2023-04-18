import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the data from the CSV file
data = pd.read_csv('HIGGS_train.csv',low_memory= False)
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
# Preprocess the data by splitting into features and labels, and splitting into training and test sets
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the decision tree classifier model
clf = DecisionTreeClassifier(max_depth=60, random_state=42)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the performance of the model using accuracy as a metric
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
