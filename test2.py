import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utis import load_dataset
from utis import *

from dlFramework import frame as fw

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]


index = 20
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
plt.show()



# turn images into numpy array (64*64*3,M) here M is number of examples
# first it is (M,64*64*3) then Transpose so it become (64*64*3,M)

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


model = fw.MultiLayer()

model.addLayerInput(train_set_x.shape[0])

model.addHidenLayer(20,act_func=fw.relu)

model.addHidenLayer(7,act_func=fw.relu)

model.addHidenLayer(5,act_func=fw.relu)

# model.addHidenLayer(128,act_func=fw.sigmoid)
#
# model.addHidenLayer(64,act_func=fw.sigmoid)
#
# model.addHidenLayer(32,act_func=fw.sigmoid)

model.addOutputLayer(train_set_y.shape[0],act_func=fw.sigmoid)

model.initialize_parameters(seed=1)

#compare between batch and mini batch make batch of 20 and iter = 220 and batch of 0 and iter 2500 and add drop out of 0.7
parameters,costs = model.train(train_set_x,train_set_y,num_iterations=220,print_cost=True , cont=0 ,learning_rate=0.0075 ,batch_size=20,print_cost_each=10,opt_func=fw.gd_optm,reg_term=0,drop=0 )

#print(parameters)

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()

acc = model.test(train_set_x,train_set_y)

print(acc)

acc = model.test(test_set_x,test_set_y)

print(acc)

