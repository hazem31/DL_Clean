import numpy as np
import matplotlib.pyplot as plt
from num2.testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from num2.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from dlFramework import frame as fw

np.random.seed(1)

X, Y = load_planar_dataset()

plt.scatter(X[0, :], X[1, :], c=Y[0,:], s=40, cmap=plt.cm.Spectral);
plt.show()


model = fw.MultiLayer()

model.addLayerInput(X.shape[0])

model.addHidenLayer(4,act_func=fw.sigmoid)


model.addOutputLayer(Y.shape[0],act_func=fw.sigmoid)

model.initialize_parameters(seed=2)

parameters = model.train(X,Y,num_iterations=10000,print_cost=True , cont=1)

print(parameters)

acc = model.test(X,Y)

print(acc)

plot_decision_boundary(lambda x: (model.predict(x.T) > 0.5) * 1, X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()


# X_assess, Y_assess = nn_model_test_case()
# plt.scatter(X_assess[0, :], X_assess[1, :], c=Y_assess[0,:], s=40, cmap=plt.cm.Spectral);
# plt.show()
#
#
# model = fw.MultiLayer()
#
# model.addLayerInput(X_assess.shape[0])
#
#
# model.addOutputLayer(Y_assess.shape[0],act_func=fw.sigmoid)
#
# model.initialize_parameters(seed=1)
#
# parameters = model.train(X_assess,Y_assess,num_iterations=10000,print_cost=True , cont=1)
#
# print(parameters)
#
# plot_decision_boundary(lambda x: (model.predict(x.T) > 0.5) * 1, X_assess, Y_assess)
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()