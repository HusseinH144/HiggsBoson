from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np

def get_model():
    model = load_model('model.h5')
    return model
# Load the data Here Replace file Name
df = pd.read_csv('HIGGS_train.csv',low_memory=False)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
# Split the data into features and labels
X = df.iloc[:, 1:]
y = df.iloc[:,0]

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
model = get_model()
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