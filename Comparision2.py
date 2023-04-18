import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from keras.optimizers import Adam


# Load the data
df = pd.read_csv('HIGGS_train.csv',low_memory=False)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
# Split the data into features and labels
X = df.iloc[:100000, 1:]
y = df.iloc[:100000,0]

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create a neural network with two hidden layers
model = Sequential()
model.add(Dense(256, input_dim=28, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Compile the neural network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the neural network
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)
# Evaluate the neural network classifier
acc_mlp = accuracy_score(y_test, y_pred)
f1_mlp = f1_score(y_test, y_pred)
prec_mlp = precision_score(y_test, y_pred)

print("Neural Network Classifier Results:")
print("Accuracy:", acc_mlp)
print("F1 Score:", f1_mlp)
print("Precision:", prec_mlp)

# # Train a histogram gradient boosting classifier
# hgbc = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.01, max_depth=30, min_samples_leaf=20, random_state=42)
# hgbc.fit(X_train, y_train)
#
# # Evaluate the histogram gradient boosting classifier
# y_pred_hgbc = hgbc.predict(X_test)
# acc_hgbc = accuracy_score(y_test, y_pred_hgbc)
# f1_hgbc = f1_score(y_test, y_pred_hgbc)
# prec_hgbc = precision_score(y_test, y_pred_hgbc)
#
# print("Histogram Gradient Boosting Classifier Results:")
# print("Accuracy:", acc_hgbc)
# print("F1 Score:", f1_hgbc)
# print("Precision:", prec_hgbc)
