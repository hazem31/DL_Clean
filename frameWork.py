import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def identity(z):
    return z

def initialize_with_zeros(dim):

    w = np.zeros(shape=dim)
    b = np.zeros(shape=dim[0])
    # assert (w.shape == dim)
    # assert (isinstance(b, float) or isinstance(b, int))
    return w, b

def logLikehood_cost_grad(m, Y, A, X):
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    dw = (1 / m) * np.dot(X, (A - Y).T).T
    db = (1 / m) * np.sum(A - Y).T
    return cost, dw, db


def optimize_sgd(model,X, Y, num_iterations, learning_rate, print_cost=False , epsilion = 0.0001):
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

    def __init__(self, number_of_neurons, number_of_outputs=1, act_func=identity, init_func=initialize_with_zeros, cost_func=logLikehood_cost_grad):
        self.w, self.b = init_func((number_of_outputs, number_of_neurons))
        self.number_of_neurons = number_of_neurons
        self.number_of_outputs = number_of_outputs
        self.act_func = act_func
        self.cost_func = cost_func

        if number_of_outputs == 1:
            self.classes = 2
        else:
            self.classes = number_of_outputs


    def re_init(self,init_func):
        self.w ,self.b = init_func((self.number_of_outputs,self.number_of_neurons))


    def propagate(self, X, Y , type_of_y = '0'):
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
        A =self.act_func(Z)

        cost, dw, db = self.cost_func(m, Y, A,X)

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def predict(self,X,threshold = 0.5,z_value=False):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        #todo should a class start with a one or zero for the first class
        #num_of_classes = [i for i in range(self.classes)]
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        #w = w.reshape(X.shape[0], 1)

        Z = np.dot(self.w , X) + self.b

        if z_value == True:
            return Z

        A = self.act_func(Z)

        if self.classes == 2:
            for i in range(A.shape[1]):
                #todo check if should be 0 or -1
                Y_prediction[0, i] = 1 if A[0, i] > threshold else 0

        else:
            for i in range(A.shape[1]):
                #todo check this later
                Y_prediction[0, i] = np.argmax(A[:,i])

        return Y_prediction

    def train(self ,X_train , Y_train,num_iterations=2000, learning_rate=0.5, print_cost=False , optmizer=optimize_sgd):
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
        grads , costs = optmizer(self, X_train, Y_train, num_iterations, learning_rate, print_cost)

        # Predict test/train set examples (≈ 2 lines of code)

        #Y_prediction_test = self.predict(X_test)
        #Y_prediction_train = self.predict(X_train)



        #print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        #print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "w": self.w,
             "b": self.b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return d

    def accuracy(self,X,Y):
        prediction = self.predict(X)
        accuracy = 100 - np.mean(np.abs(prediction - Y)) * 100
        return accuracy

    def test(self,X_test,Y_test):
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


