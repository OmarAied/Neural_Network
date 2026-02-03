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
X_dev = data_dev[1:n] / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.0

# Normalize the initial weights and biases using He initialization
def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(2.0 / 784)
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * np.sqrt(2.0 / 10)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

# Activation functions
def Relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(np.exp(Z), axis=0, keepdims=True)

# Forward propagation to compute activations
def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X) + b1
    A1 = Relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# One-hot encoding of labels
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Derivative of ReLU activation
def deriv_relu(Z):
    return Z > 0

# Backward propagation to compute gradients
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / Y.size) * dZ2.dot(A1.T)
    db2 = (1 / Y.size) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = (1 / Y.size) * dZ1.dot(X.T)
    db1 = (1 / Y.size) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update weight and bias parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Get predictions from output activations
def get_predictions(A2):
    return np.argmax(A2, 0)

# Calculate accuracy of predictions
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent to train the model
def gradient_descent(X, Y, iterations, learning_rate):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

# Train the model
print("Training the model...")
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

# Evaluate the model on the development set
print("Evaluating on development set...")
Z1_dev, A1_dev, Z2_dev, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
dev_accuracy = get_accuracy(get_predictions(A2_dev), Y_dev)
print("Development set accuracy: ", dev_accuracy)
