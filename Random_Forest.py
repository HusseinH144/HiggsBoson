import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score

# Load the data from the CSV file
data = pd.read_csv('HIGGS_train.csv',low_memory=False)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()
# Split the data into features (X) and labels (y)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier with n_estimators trees
rf = RandomForestClassifier(n_estimators=500, random_state=42)

# Train the random forest classifier
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_pred = np.round(y_pred)
# Evaluate the neural network classifier
acc_mlp = accuracy_score(y_test, y_pred)
f1_mlp = f1_score(y_test, y_pred)
prec_mlp = precision_score(y_test, y_pred)

print("Random Forest Classifier Results:")
print("Accuracy:", acc_mlp)
print("F1 Score:", f1_mlp)
print("Precision:", prec_mlp)
