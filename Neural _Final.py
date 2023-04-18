import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
from keras import regularizers

# Load the data
df = pd.read_csv('HIGGS_train.csv',low_memory=False)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
# Split the data into features and labels
X = df.iloc[:, 1:]
y = df.iloc[:,0]

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a neural network with two hidden layers
model = Sequential()
model.add(Dense(256, input_dim=28, activation='relu'))

model.add(Dense(128, activation='relu',))

model.add(Dense(128, activation='sigmoid',))

model.add(Dense(128, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))
# Compile the neural network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=64, verbose=1)
y_pred = model.predict(X)
y_pred = np.round(y_pred)
# Evaluate the neural network classifier
acc_mlp = accuracy_score(y, y_pred)
f1_mlp = f1_score(y, y_pred)
prec_mlp = precision_score(y, y_pred)

print("Neural Network Classifier Results:")
print("Accuracy:", acc_mlp)
print("F1 Score:", f1_mlp)
print("Precision:", prec_mlp)
model.save('model.h5')
