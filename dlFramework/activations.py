import numpy as np

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def identity(z):
    return z

def identity_der(z):
    return 1


def sigmoid_der(A):
    return A * (1 - A)


def tanh(z):
    return np.tanh(z)


def tanh_der(A):
    return 1 - A ** 2


# added in colab
def softmax(z):
    # print("in softmax")
    return np.exp(z) / (np.exp(z).sum(axis=0))


# added in colab
def softmax_der(A):
    # print("in softmax der")
    # temporaryly
    return 1


def relu(z):
    A = np.maximum(0, z)
    return A


def relu_der(A):
    Z = np.array(A, copy=True)
    der = np.array(A, copy=True)

    der[Z <= 0] = 0
    der[Z > 0] = 1

    return der

def determine_der_act_func(func):
    if func == sigmoid:
        return sigmoid_der
    elif func == tanh:
        return tanh_der
    elif func == relu:
        return relu_der
    elif func == softmax:
        return softmax_der
    elif func == identity:
        return identity_der

