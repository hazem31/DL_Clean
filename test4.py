#important
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

import dlFramework.frame as fw

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


train_X, train_Y, test_X, test_Y = load_2D_dataset()
plt.show()

print(train_X.shape)


model = fw.MultiLayer()

model.addLayerInput(train_X.shape[0])

model.addHidenLayer(20,act_func=fw.relu)

model.addHidenLayer(3,act_func=fw.relu)

model.addOutputLayer(train_Y.shape[0],act_func=fw.sigmoid)

model.initialize_parameters(seed=1)

parm = {}

parm['beta'] = 0.98
#make drop out 0.8 , and reg = 0.01 compare of regulrization
parameters,costs = model.train(train_X,train_Y,num_iterations=10000,print_cost=True , cont=0 ,learning_rate=0.03 ,batch_size=20,print_cost_each=100,opt_func=fw.gd_optm,param_dic=parm,reg_term=0.01,drop=0 )

#print(parameters)

acc = model.test(train_X,train_Y)

print(acc)

acc = model.test(test_X,test_Y)

print(acc)


plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()


plot_decision_boundary(lambda x: (model.predict(x.T) > 0.5) * 1, train_X, train_Y)
plt.show()