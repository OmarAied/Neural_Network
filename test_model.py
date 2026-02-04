import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Activation functions
def Relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Forward propagation to compute activations
def forward_prop(w1,b1,w2,b2,w3,b3,X):
    # Layer 1
    Z1 = w1.dot(X) + b1
    A1 = Relu(Z1)
    # Layer 2
    Z2 = w2.dot(A1) + b2
    A2 = Relu(Z2)
    # Layer 3
    Z3 = w3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Get predictions from output activations
def get_predictions(A3):
    return np.argmax(A3, 0)

# Calculate accuracy of predictions
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Load model parameters from file
def load_model(filename='trained_model.pkl'):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params['w1'], params['b1'], params['w2'], params['b2'], params['w3'], params['b3']

