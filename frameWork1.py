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
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

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


class OneLayer:
    def __init__(self, number_of_neurons, number_of_outputs=1, act_func=identity, init_func=initialize_with_zeros,
                 cost_func=logLikehood_cost_grad):
        self.w, self.b = init_func((number_of_outputs, number_of_neurons))
        self.number_of_neurons = number_of_neurons
        self.number_of_outputs = number_of_outputs
        self.act_func = act_func
        self.cost_func = cost_func

        if number_of_outputs == 1:
            self.classes = 2
        else:
            self.classes = number_of_outputs

    def re_init(self, init_func):
        self.w, self.b = init_func((self.number_of_outputs, self.number_of_neurons))

    def propagate(self, X, Y, type_of_y='0'):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        Tips:
        - Write your code step by step for the propagation
        """
        m = X.shape[1]

        Z = np.dot(self.w, X) + self.b
        A = self.act_func(Z)

        cost, dw, db = self.cost_func(m, Y, A, X)

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def predict(self, X, threshold=0.5, z_value=False):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        # todo should a class start with a one or zero for the first class
        # num_of_classes = [i for i in range(self.classes)]
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        # w = w.reshape(X.shape[0], 1)

        Z = np.dot(self.w, X) + self.b

        if z_value == True:
            return Z

        A = self.act_func(Z)

        if self.classes == 2:
            for i in range(A.shape[1]):
                # todo check if should be 0 or -1
                Y_prediction[0, i] = 1 if A[0, i] > threshold else 0

        else:
            for i in range(A.shape[1]):
                # todo check this later
                Y_prediction[0, i] = np.argmax(A[:, i])

        return Y_prediction

    def train(self, X_train, Y_train, num_iterations=2000, learning_rate=0.5, print_cost=False):
        """
        Builds the logistic regression model by calling the function you've implemented previously

        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """

        # Gradient descent (≈ 1 line of code)
        grads, costs = optimize_sgd(self, X_train, Y_train, num_iterations, learning_rate, print_cost)

        # Predict test/train set examples (≈ 2 lines of code)

        # Y_prediction_test = self.predict(X_test)
        # Y_prediction_train = self.predict(X_train)



        # print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        # print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "w": self.w,
             "b": self.b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return d

    def accuracy(self, X, Y):
        prediction = self.predict(X)
        accuracy = 100 - np.mean(np.abs(prediction - Y)) * 100
        return accuracy

    def test(self, X_test, Y_test):
        Y_prediction_test = self.predict(X_test)
        accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
        return accuracy


# model = OneLayer(2, act_func=sigmoid)
# X, Y = np.array([[1,2], [3,4]]), np.array([[1, 0]])
# model.w = np.array([[1], [2]]).T
# model.b = 2
# grad ,cost = model.propagate(X, Y)
#
# print(cost)
# print(grad)
#
# print(model.predict(X))
#
# grads, costs = optimize_sgd(model, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
#
# print("w = " + str(model.w))
# print("b = " + str(model.b))
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))

def cross_entropy(m, A, Y):
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    return cost


def cross_entropy_der(m, A, Y):
    return ((-1 * Y) / A) + ((1 - Y) / (1 - A))


def sigmoid_der(A):
    return A * (1 - A)


def tanh(z):
    return np.tanh(z)


def tanh_der(A):
    return 1 - A ** 2



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

def determine_der_cost_func(func):
    if func == cross_entropy:
        return cross_entropy_der


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
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        #todo very important check later

        np.random.seed(seed)  # we set up a seed so that your output matches ours although the initialization is random.

        # for i in range(len(self.layer_size) - 1):
        #     out = init_func(self.layer_size[i + 1], self.layer_size[i])
        #     self.w.append(out[0])
        #     self.b.append(out[1])
        L = len(self.layer_size)  # number of layers in the network

        for l in range(1, L):
            self.w.append(np.random.randn(self.layer_size[l], self.layer_size[l - 1]) / np.sqrt(
                self.layer_size[l - 1]))  # *0.01
            self.b.append(np.zeros((self.layer_size[l], 1)))

        for i in range(len(self.layer_size)-1):
            self.parameters["W" + str(i + 1)] = self.w[i]
            self.parameters["b" + str(i + 1)] = self.b[i]

        return self.parameters

    def forward_propagation(self, X):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
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
        """
        Implement the backward propagation using the instructions above.

        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]

        # just for testing
        # temp = []
        # if self.prev[0][0] != 1:
        #     temp.append((1, X))
        #     for i in range(len(self.prev)):
        #         temp.append(self.prev[i])
        #
        # self.prev = temp

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
            A = self.prev[i-1][1]
            dwi = (1 / m) * np.dot(dzi, self.prev[i-1][1].T)
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

    def set_cashe(self, cache,X):
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
            self.parameters["W"+str(i+1)] = self.w[i]
            self.parameters["b" + str(i + 1)] = self.b[i]

    def update_parameters(self,grads, learning_rate=1.2):
        """
        Updates parameters using the gradient descent update rule given above

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
        # Retrieve each parameter from the dictionary "parameters"


        # Retrieve each gradient from the dictionary "grads"
        ### START CODE HERE ### (≈ 4 lines of code)

        for i in range(len(self.w)):
            self.w[i] = self.w[i] - learning_rate * grads["dW"+str(i+1)]
            self.b[i] = self.b[i] - learning_rate * grads["db" + str(i+1)]



        self.set_parameters_internal()

        return self.parameters

    def train(self,X, Y, num_iterations=10000, print_cost=False, init_func=random_init_zero_bias ,cont=0 ,learning_rate=1):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        # if cont == 0:
        #     self.initialize_parameters(init_func=init_func,seed=3)
        #     print(self.w)

        for i in range(0, num_iterations):

            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            Alast, cache = self.forward_propagation(X)

            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = self.compute_cost(Alast, Y)

            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = self.backward_propagation(X, Y)

            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = self.update_parameters(grads,learning_rate=learning_rate)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters