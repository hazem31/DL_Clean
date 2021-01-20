import numpy as np
from dlFramework.loss import *
from dlFramework.activations import *
from dlFramework.eval import *
from dlFramework.optmizers import *
from dlFramework.data import *


class MultiLayer:
    def __init__(self, number_of_neurons=0, cost_func=cross_entropy):
        self.w, self.b = [], []
        self.parameters = {}
        self.layer_size = []

        self.number_of_input_neurons = number_of_neurons
        self.number_of_outputs = 0

        self.act_func = []
        self.derivative_act_func = []

        self.cost_func = cost_func
        self.cost_func_der = determine_der_cost_func(self.cost_func)

        self.cache = {}
        self.prev = []

    def addLayerInput(self, size):
        self.number_of_input_neurons = size
        self.layer_size.append(size)

    def addHidenLayer(self, size, act_func=sigmoid):
        self.layer_size.append(size)
        self.act_func.append(act_func)
        self.derivative_act_func.append(determine_der_act_func(act_func))

    def addOutputLayer(self, size, act_func=sigmoid):
        self.number_of_outputs = size
        self.layer_size.append(size)
        self.act_func.append(act_func)
        self.derivative_act_func.append(determine_der_act_func(act_func))

    def initialize_parameters(self, seed=2): #,init_func=random_init_zero_bias):

        # todo very important check later

        np.random.seed(seed)  # we set up a seed so that your output matches ours although the initialization is random.

        L = len(self.layer_size)  # number of layers in the network

        for l in range(1, L):
            self.w.append(np.random.randn(self.layer_size[l], self.layer_size[l - 1]) * np.sqrt
            (2 / self.layer_size[l - 1]))  # *0.01
            self.b.append(np.zeros((self.layer_size[l], 1)))
            # seed += 1
            # np.random.seed(seed)

        for i in range(len(self.layer_size) - 1):
            self.parameters["W" + str(i + 1)] = self.w[i]
            self.parameters["b" + str(i + 1)] = self.b[i]

        return self.parameters

    def forward_propagation(self, X,drop=0):

        self.prev = []
        self.prev.append((1, X))
        for i in range(len(self.layer_size) - 1):
            Zi = np.dot(self.w[i], self.prev[i][1]) + self.b[i]
            Ai = self.act_func[i](Zi)
            if drop > 0 and i != len(self.layer_size) - 2:
                D = np.random.rand(Ai.shape[0], Ai.shape[1])
                D = D < drop
                Ai = Ai * D
                Ai = Ai / drop

            self.prev.append((Zi, Ai))

        A_last = self.prev[-1][1]

        for i in range(len(self.layer_size) - 1):
            self.cache["Z" + str(i + 1)] = self.prev[i + 1][0]
            self.cache["A" + str(i + 1)] = self.prev[i + 1][1]

        # todo sould i compute cost in here

        return A_last, self.cache

    def set_cost(self, cost_func):
        self.cost_func = cost_func
        self.cost_func_der = determine_der_cost_func(cost_func)

    def compute_cost(self, Alast, Y):
        m = Alast.shape[1]
        return self.cost_func(m, Alast, Y)

    def backward_propagation(self, X, Y):

        m = X.shape[1]

        # todo all depends on the type of function in cost and actviation function
        grad_list1_w = []
        grad_list1_b = []

        Alast = self.prev[-1][1]
        final_act = self.derivative_act_func[-1]
        dzi = self.cost_func_der(m, Alast, Y) * final_act(Alast)

        if self.cost_func == cross_entropy:
            if self.act_func[-1] == sigmoid:
                pass

        for i in range(len(self.w), 0, -1):
            A = self.prev[i - 1][1]
            dwi = (1 / m) * np.dot(dzi, self.prev[i - 1][1].T)
            dbi = (1 / m) * np.sum(dzi, axis=1, keepdims=True)
            if i != 1:
                der_func = self.derivative_act_func[i - 2]
                A = self.prev[i - 1][1]
                dzi = np.multiply(np.dot((self.w[i - 1]).T, dzi), der_func(A))

            grad_list1_w.append(dwi)
            grad_list1_b.append(dbi)

        # reverse grad list
        grad_list_w = []
        grad_list_b = []

        for i in range(len(grad_list1_w) - 1, -1, -1):
            grad_list_w.append(grad_list1_w[i])
            grad_list_b.append(grad_list1_b[i])

        grads = {}

        for i in range(len(grad_list_w)):
            grads['dW' + str(i + 1)] = grad_list_w[i]
            grads['db' + str(i + 1)] = grad_list_b[i]

        return grads

    def set_cashe(self, cache, X):
        self.cache = cache
        self.prev = []
        self.prev.append((1, X))
        for i in range(int(len(cache.keys()) / 2)):
            A, Z = cache["A" + str(i + 1)], cache["Z" + str(i + 1)]
            self.prev.append((Z, A))

    def set_parameters(self, para):
        self.parameters = para
        self.w = []
        self.b = []
        for i in range(int(len(para.keys()) / 2)):
            W, b = para["W" + str(i + 1)], para["b" + str(i + 1)]
            self.w.append(W)
            self.b.append(b)

    def set_parameters_internal(self):
        self.parameters = {}
        for i in range(len(self.w)):
            self.parameters["W" + str(i + 1)] = self.w[i]
            self.parameters["b" + str(i + 1)] = self.b[i]

    def update_parameters(self, grads, learning_rate=1.2 , reg_term=0, m = 1):

        for i in range(len(self.w)):
            self.w[i] = (1-(reg_term/m)) * self.w[i] - learning_rate * grads["dW" + str(i + 1)]
            self.b[i] = (1-(reg_term/m)) * self.b[i] - learning_rate * grads["db" + str(i + 1)]

        self.set_parameters_internal()

        return self.parameters

    def train(self, X, Y, num_iterations=10000, print_cost=False , print_cost_each=100, cont=0, learning_rate=1 , reg_term=0 , batch_size=0 , opt_func=gd_optm, param_dic=None,drop=0):


        parameters , costs = opt_func(self,X,Y , num_iterations,print_cost,print_cost_each,cont,learning_rate,reg_term,batch_size , param_dic,drop)
        return parameters , costs

    def predict(self, X):

        Alast, cache = self.forward_propagation(X)
        #predictions = (Alast > thres) * 1

        return Alast

    def test(self, X, Y,eval_func=accuracy_score):


        Alast, cache = self.forward_propagation(X)

        acc = eval_func(Alast,Y)

        return acc

