import numpy as np


def accuracy_score(A,Y,thres=0.5):

    if Y.shape[0] == 1:
        predictions = (A > thres) * 1
        acc = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size))
    else:
        index = A.argmax(axis=0)
        yi = Y.argmax(axis=0)
        acc = ((np.sum(index == yi)) / yi.shape[0]) * 1.0

    return acc * 100

def precion_score(A,Y,thres=0.5):
    pass

def recal_score(A,Y,thres=0.5):
    pass

def f1_score(A,Y,thres=0.5):
    pass
