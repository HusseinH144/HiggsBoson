from keras import regularizers
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, precision_score
# Load the data
df = pd.read_csv('HIGGS_train.csv',low_memory=False)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split the data into features and labels
X = df.iloc[:, 1:]
y = df.iloc[:,0]
# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a neural network with two hidden layers and L2 regularization
model = Sequential()
model.add(Dense(64, input_dim=28, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1, activation='sigmoid'))

# Compile the neural network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the neural network
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1, validation_data=(X_test, y_test))
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