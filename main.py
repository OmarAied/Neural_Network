import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the MNIST dataset
data = pd.read_csv('mnist_train.csv')
data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

# Split the dataset into development and training sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def Relu(Z):
    return np.maximum(0, Z)

def forward_prop(w1,b1,w2,b2,X):
    pass