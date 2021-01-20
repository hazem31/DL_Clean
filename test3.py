import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, predict_dec
from dlFramework import frame as fw
from num2.planar_utils import plot_decision_boundary

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()
plt.show()


model = fw.MultiLayer()

model.addLayerInput(train_X.shape[0])

model.addHidenLayer(4,act_func=fw.sigmoid)


model.addOutputLayer(train_Y.shape[0],act_func=fw.sigmoid)

model.initialize_parameters(seed=2)

parm = {}
parm['beta'] = 0.98

#comapre between gd and mom_optm
parameters , costs = model.train(train_X,train_Y,num_iterations=10000,print_cost=True , print_cost_each=1 , cont=1,opt_func=fw.mom_optm,param_dic=parm)

#print(parameters)

print(model.test(train_X,train_Y))
print(model.test(test_X,test_Y))

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()



plot_decision_boundary(lambda x: (model.predict(x.T) > 0.5) * 1, train_X, train_Y)
plt.show()