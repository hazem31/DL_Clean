import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def identity(z):
    return z


def initialize_with_zeros(dim):
    w = np.zeros(shape=dim)
    b = np.zeros(shape=(dim[0], 1))
    # assert (w.shape == dim)
    # assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def logLikehood_cost_grad(m, Y, A, X):
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    dw = (1 / m) * np.dot(X, (A - Y).T).T
    db = (1 / m) * np.sum(A - Y).T
    return cost, dw, db


def optimize_sgd(model, X, Y, num_iterations, learning_rate, print_cost=False, epsilion=0.0001):
    costs = []

    for i in range(num_iterations):

        grads, cost = model.propagate(X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        model.w = model.w - learning_rate * dw  # need to broadcast
        model.b = model.b - learning_rate * db

        costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    grads = {"dw": dw,
             "db": db}

    return grads, costs


def cross_entropy(m, A, Y):
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    return cost


def cross_entropy_der(m, A, Y):
    return ((-1 * Y) / A) + ((1 - Y) / (1 - A))


# added in colab
def cross_multi_class(m, A, Y):
    # v1 = Y * A
    # v2 = np.max(v1,axis=0)
    # v3 = np.log(v2).sum()
    # return (-1 / m) * v3

    cost = (-1 / m) * np.sum((Y) * (np.log(A)))
    # print("in cross multi")
    return cost


# added in colab
def cross_multi_class_der(m, A, Y):
    z1 = np.array(A, copy=True)
    y1 = np.array(Y, copy=True)
    y1[y1 == 1] = -1
    return A - Y


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


def random_init_zero_bias(n_2, n_1, mult=0.01):
    return np.random.randn(n_2, n_1) * 0.01, np.zeros(shape=(n_2, 1))


def determine_der_act_func(func):
    if func == sigmoid:
        return sigmoid_der
    elif func == tanh:
        return tanh_der
    elif func == relu:
        return relu_der
    elif func == softmax:
        return softmax_der


def determine_der_cost_func(func):
    if func == cross_entropy:
        return cross_entropy_der
    if func == cross_multi_class:
        return cross_multi_class_der


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

    def initialize_parameters(self, seed=2, init_func=random_init_zero_bias):


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

    def forward_propagation(self, X):

        self.prev = []
        self.prev.append((1, X))
        for i in range(len(self.layer_size) - 1):
            Zi = np.dot(self.w[i], self.prev[i][1]) + self.b[i]
            Ai = self.act_func[i](Zi)
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

    def update_parameters(self, grads, learning_rate=1.2):

        for i in range(len(self.w)):
            self.w[i] = self.w[i] - learning_rate * grads["dW" + str(i + 1)]
            self.b[i] = self.b[i] - learning_rate * grads["db" + str(i + 1)]

        self.set_parameters_internal()

        return self.parameters

    def train(self, X, Y, num_iterations=10000, print_cost=False, init_func=random_init_zero_bias, cont=0,
              learning_rate=1):

        for i in range(0, num_iterations):

            Alast, cache = self.forward_propagation(X)

            cost = self.compute_cost(Alast, Y)

            grads = self.backward_propagation(X, Y)

            parameters = self.update_parameters(grads, learning_rate=learning_rate)

            if print_cost and i % 1 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters

    def test(self, X, Y):

        Alast, cache = self.forward_propagation(X)

        index = Alast.argmax(axis=0)
        yi = Y.argmax(axis=0)

        return ((np.sum(index == yi)) / yi.shape[0]) * 1.0


        # added in colab for batch gradient

        # for i in range(0, num_iterations):
        #           for j in range(int(X.shape[1]/100)):

        #           # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        #               Alast, cache = self.forward_propagation(X[:,j:j+100])

        #           # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        #               cost = self.compute_cost(Alast, Y[:,j:j+100])

        #           # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        #               grads = self.backward_propagation(X[:,j:j+100], Y[:,j:j+100])

        #           # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        #               parameters = self.update_parameters(grads,learning_rate=learning_rate)

        #           if print_cost and i % 2 == 0:
        #               print("Cost after iteration %i: %f" % (i, cost))
