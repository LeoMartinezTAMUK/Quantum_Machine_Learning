# Basic Quantum Machine Learning (QML) Implementation on Synthetic Data
# Created by: Leo Martinez III in Fall 2024

# Imports
import pandas as pd
import numpy as np
from pennylane import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

import pennylane as qml # QML Python Library
from pennylane.templates.embeddings import AngleEmbedding, AmplitudeEmbedding
from pennylane.optimize import AdamOptimizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#%%----------------------------------------------------------------------------

# Read out CSV and sets/samples creation
np.random.seed(30) # to easily be reproduced
df = pd.read_csv(r'data/synthetic_data.csv', sep=',')
df = df.astype(float)
train, test = train_test_split(df, test_size=0.30, random_state=2)

# Subset of the dataset
#train_set = train.sample(160)
#test_set = test.sample(40)

# Uncomment to use the entire dataset
train_set = train
test_set = test

#%%----------------------------------------------------------------------------

# Separation of labels
x_train = train_set
y_train = train_set[['y']]
x_test = test_set
y_test = test_set[['y']]

#%%----------------------------------------------------------------------------

# Reduce dimensions using PCA so later you can fit the dimensions with the qubits
n_dim = 2
pca = PCA(n_components=n_dim)

pca.fit(x_train)
x_train = pca.transform(x_train)

pca.fit(x_test)
x_test = pca.transform(x_test)

#%%----------------------------------------------------------------------------

# Standardize both training and test data using the same scaler
std_scale = StandardScaler().fit(x_train)
x_train = std_scale.transform(x_train)
x_test = std_scale.transform(x_test)

#%%----------------------------------------------------------------------------

# Assignment
X = np.array(x_train, requires_grad=False)
Xte = np.array(x_test, requires_grad=False)

#%%----------------------------------------------------------------------------

# Shift label from {0, 1} to {-1, 1}
Y = np.array(y_train.values[:,0] * 2 - np.ones(len(y_train.values[:,0])), requires_grad = False)
Yte = np.array(y_test.values[:,0] * 2 - np.ones(len(y_test.values[:,0])), requires_grad = False)

#%%----------------------------------------------------------------------------

# Angle Encoding
num_qubits = n_dim
dev = qml.device('default.qubit', wires=num_qubits)

@qml.qnode(dev)
def circuit(parameters, data):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    AngleEmbedding(features=data, wires=range(num_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))

#%%----------------------------------------------------------------------------

# Initialize the parameters
num_layers = 5
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)

#%%----------------------------------------------------------------------------

# Initialization
opt = AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.99, eps=1e-08)
batch_size = 10

weights = weights_init
bias = bias_init

wbest = weights
bbest = bias
abest = accuracy(Y, [np.sign(variational_classifier(weights, bias, x)) for x in X])

for it in range(5):
    batch_index = np.random.randint(0, len(X), (batch_size,))
    X_batch = X[batch_index]
    Y_batch = Y[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]

    current_accuracy = accuracy(Y, predictions)
    if current_accuracy > abest:
        wbest = weights
        bbest = bias
        abest = current_accuracy
        print('New best')

    print(f"Iter: {it + 1} | Cost: {cost(weights, bias, X, Y):.7f} | Accuracy: {current_accuracy:.7f}")
    
#%%----------------------------------------------------------------------------

# Perform binary classification on the dataset and print results
predictions = [np.sign(variational_classifier(wbest, bbest, x)) for x in Xte]
acc = accuracy(Yte, predictions)

print(f'Cost: {cost(wbest, bbest, Xte, Yte)}, Accuracy: {np.round(acc, 2) * 100}%')

# Print the classification report and important metrics
print(metrics.classification_report(Yte, predictions))
print(metrics.precision_score(Yte, predictions))
print(metrics.recall_score(Yte, predictions))
print(metrics.f1_score(Yte, predictions))
print(metrics.balanced_accuracy_score(Yte, predictions))