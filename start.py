import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utis import load_dataset
import frameWork1 as fw
from utis import *

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


#load data

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# print("Number of training examples: m_train = " + str(m_train))
# print("Number of testing examples: m_test = " + str(m_test))
# print("Height/Width of each image: num_px = " + str(num_px))
# print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print("train_set_x shape: " + str(train_set_x_orig.shape))
# print("train_set_y shape: " + str(train_set_y.shape))
# print("test_set_x shape: " + str(test_set_x_orig.shape))
# print("test_set_y shape: " + str(test_set_y.shape))

# expected output
# Number of training examples: m_train = 209
# Number of testing examples: m_test = 50
# Height/Width of each image: num_px = 64
# Each image is of size: (64, 64, 3)
# train_set_x shape: (209, 64, 64, 3)
# train_set_y shape: (1, 209)
# test_set_x shape: (50, 64, 64, 3)
# test_set_y shape: (1, 50)

# index = 20
# plt.imshow(train_set_x_orig[index])
# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
# plt.show()



# turn images into numpy array (64*64*3,M) here M is number of examples
# first it is (M,64*64*3) then Transpose so it become (64*64*3,M)

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
### END CODE HERE ###

# print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print("train_set_y shape: " + str(train_set_y.shape))
# print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print("test_set_y shape: " + str(test_set_y.shape))
# print("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# expected output
# train_set_x_flatten shape: (12288, 209)
# train_set_y shape: (1, 209)
# test_set_x_flatten shape: (12288, 50)
# test_set_y shape: (1, 50)
# sanity check after reshaping: [17 31 56 22 33]

# Normalize data

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# model = fw.OneLayer(train_set_x.shape[0] , number_of_outputs=1 , act_func=fw.sigmoid,init_func=fw.initialize_with_zeros, cost_func=fw.logLikehood_cost_grad)
#
# d = model.train(train_set_x,train_set_y,num_iterations=2000,learning_rate=0.005,print_cost=False)
#
#
#
# print(model.accuracy(train_set_x,train_set_y))
# print(model.accuracy(test_set_x,test_set_y))

# print(model.b.shape)

# costs = np.squeeze(d['costs'])
# print(costs)
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()

model = fw.MultiLayer()

model.addLayerInput(train_set_x.shape[0])

model.addHidenLayer(20,act_func=fw.relu)

model.addHidenLayer(7,act_func=fw.relu)

model.addHidenLayer(5,act_func=fw.relu)

model.addOutputLayer(train_set_y.shape[0],act_func=fw.sigmoid)

model.initialize_parameters(seed=1)

parameters = model.train(train_set_x,train_set_y,num_iterations=2500,print_cost=True , cont=0 ,learning_rate=0.0075 , init_func=fw.initialize_with_zeros)

print(parameters)