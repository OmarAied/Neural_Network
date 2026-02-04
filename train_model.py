import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# Normalize the initial weights and biases using He initialization
def init_params():
    # Layer 1
    w1 = np.random.randn(128, 784) * np.sqrt(2.0 / 784)
    b1 = np.zeros((128, 1))
    # Layer 2
    w2 = np.random.randn(64, 128) * np.sqrt(2.0 / 128)
    b2 = np.zeros((64, 1))
    # Layer 3
    w3 = np.random.randn(10, 64) * np.sqrt(2.0 / 10)
    b3 = np.zeros((10, 1))
    return w1, b1, w2, b2, w3, b3

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
def back_prop(Z1, A1, Z2, A2, A3, w2, w3, X, Y):
    one_hot_Y = one_hot(Y)
    # Layer 3
    dZ3 = A3 - one_hot_Y
    dw3 = (1 / Y.size) * dZ3.dot(A2.T)
    db3 = (1 / Y.size) * np.sum(dZ3, axis=1, keepdims=True)
    # Layer 2
    dA2 = w3.T.dot(dZ3)
    dZ2 = dA2 * deriv_relu(Z2)
    dw2 = (1 / Y.size) * dZ2.dot(A1.T)
    db2 = (1 / Y.size) * np.sum(dZ2, axis=1, keepdims=True)
    # Layer 1
    dA1 = w2.T.dot(dZ2)
    dZ1 = dA1 * deriv_relu(Z1)
    dw1 = (1 / Y.size) * dZ1.dot(X.T)
    db1 = (1 / Y.size) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dw1, db1, dw2, db2, dw3, db3

# Update weight and bias parameters
def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    return w1, b1, w2, b2, w3, b3

# Get predictions from output activations
def get_predictions(A3):
    return np.argmax(A3, 0)

# Calculate accuracy of predictions
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent to train the model
def gradient_descent(X, Y, iterations, learning_rate):
    w1, b1, w2, b2, w3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, z3, A3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
        dw1, db1, dw2, db2, dw3, db3 = back_prop(Z1, A1, Z2, A2, A3, w2, w3, X, Y)
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A3), Y))
    return w1, b1, w2, b2, w3, b3


def save_model(w1, b1, w2, b2, w3, b3, filename='trained_model.pkl'):
    trained_params = {
        'w1': w1, 'b1': b1,
        'w2': w2, 'b2': b2,
        'w3': w3, 'b3': b3
    }
    with open(filename, 'wb') as f:
        pickle.dump(trained_params, f)
    print(f"Model saved to '{filename}'")
    return trained_params

def train_and_save_model():
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

    # Train the model
    print("Training the model...")
    w1, b1, w2, b2, w3, b3 = gradient_descent(X_train, Y_train, 500, 0.1)

    # Evaluate the model on the development set
    print("Evaluating on development set...")
    Z1_dev, A1_dev, Z2_dev, A2_dev, Z3_dev, A3_dev = forward_prop(w1, b1, w2, b2, w3, b3, X_dev)
    dev_accuracy = get_accuracy(get_predictions(A3_dev), Y_dev)
    print("Development set accuracy: ", dev_accuracy)

    # Save the trained model
    save_model(w1, b1, w2, b2, w3, b3)

    return w1, b1, w2, b2, w3, b3

if __name__ == "__main__":
    train_and_save_model()
