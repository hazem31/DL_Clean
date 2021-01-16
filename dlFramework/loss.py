import numpy as np


def cross_entropy(m, A, Y):
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    return cost


def cross_entropy_der(m, A, Y):
    return ((-1 * Y) / A) + ((1 - Y) / (1 - A))


def cross_multi_class(m, A, Y):
    # v1 = Y * A
    # v2 = np.max(v1,axis=0)
    # v3 = np.log(v2).sum()
    # return (-1 / m) * v3

    cost = (-1 / m) * np.sum((Y) * (np.log(A)))
    # print("in cross multi")
    return cost

def cross_multi_class_der(m, A, Y):
    z1 = np.array(A, copy=True)
    y1 = np.array(Y, copy=True)
    y1[y1 == 1] = -1
    return A - Y


def determine_der_cost_func(func):
    if func == cross_entropy:
        return cross_entropy_der
    if func == cross_multi_class:
        return cross_multi_class_der

