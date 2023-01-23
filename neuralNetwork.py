from binascii import b2a_base64
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv('/data');

data.head()

data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    w1 = np.random.rand(10 , 784)
    b1 = np.random.rand(10, 1)
    w2 = np.random.rand(10 , 10)
    b2 = np.random.rand(10, 1)
    return w1,b1,w2,b2

def ReLU(Z):
   return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(exp(Z))

def forward_prop(w1,b1,w2,b2,X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size,Y.max() + 1))
    one_hot_Y[np.arange(Y.size). Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1 , A1 , Z2, A2, W2 , Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/ m* dZ2.dot(A1.T)
    db2 = 1/ m * np.sum(dZ2, 2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/ m * dZ1.dot(X.T)
    db1 = 1/ m * np.sum(dZ1, 2)
    return dW1,db1,dW2,db2

def update_params(w1,b1,w2,b2,dW1,db1,dW2,db2,alpha):
    w1 = w1 - alpha * dW1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dW2
    b2 = b2 - alpha * db2
    return w1,b1,w2,b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions , Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X,Y, iterations, alpha):
    w1,b1,w2,b2 = init_params()
    for i in range(iterations):
        Z1,A1,Z2,A2 = forward_prop(w1,b1,w2,b2,X)
        dW1,db1,dW2,db2 = back_prop(Z1,A1,Z2,A2,w2,X,Y)
        w1,b1,w2,b2 = update_params(w1,b1,w2,b2,dW1,db1,dW2,db2,alpha)
        if i % 50 == 0:
            print("Iteration: ",i)
            print("Accuracy: ", get_accuracy(get_predictions(A2),Y))
        return w1,b1,w2,b2

