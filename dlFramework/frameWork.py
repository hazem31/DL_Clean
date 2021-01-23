import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


def sigmoid(z):
    """ This function applies sigmoid function(has mathematical form) as an activation function of a node for forwardpropagation.

    Parameters:

        z (numpy array): result of W.X (Weights.Inputs/features).

    Returns:

        (numpy array): Returning array of same size as input "z" after applying sigmoid on input.
   """
    s = 1 / (1 + np.exp(-z))
    return s


def sigmoid_der(A):
    """ This function applies sigmoid derivative function(has mathematical form) for backpropagation.

    Parameters:

        A (numpy array): result of W.X (Weights.Inputs/features).

    Returns:

        (numpy array): Returning array of same size as input "A".
   """
    return A * (1 - A)


def identity(z):
    """ This function applies identity function(has no mathematical form) as an activation function of a node for forwardpropagation.

    Parameters:

        z (numpy array): result of W.X (Weights.Inputs/features).

    Returns:

        (numpy array): Returning same array of input "z".
   """
    return z


def identity_der(z):
    """ This function applies identity derivative function(has no mathematical form) for backpropagation.

    Parameters:

        z (numpy array): result of W.X (Weights.Inputs/features).

    Returns:

        (int): Returning 1 as its derivative of z.
   """
    return 1


def tanh(z):
    """ This function applies tanh function(has mathematical form) as an activation function of a node for forwardpropagation.

    Parameters:

        z (numpy array): result of W.X (Weights.Inputs/features).

    Returns:

        (numpy array): Returning array of same size as input "z" after applying "tanh()".
   """
    return np.tanh(z)


def tanh_der(A):
    """ This function applies tanh derivative function(has mathematical form) for backpropagation.

    Parameters:

        A (numpy array): result of W.X (Weights.Inputs/features).

    Returns:

        (numpy array): Returning array of same size as input "A" after applying derivative of tanh().
   """
    return 1 - A ** 2


# added in colab
def softmax(z):
    """ This function applies softmax function(has mathematical form) as an activation function of a node for forwardpropagation.

    Parameters:

        z (numpy array): result of W.X (Weights.Inputs/features).

    Returns:

        (numpy array): Returning array of same size as input "z" after applying "exp()/sum of exponential of all inputs".
   """

    # print("in softmax")
    return np.exp(z) / (np.exp(z).sum(axis=0))


# added in colab
def softmax_der(A):
    """ This function applies softmax derivative function(has mathematical form = 1) for backpropagation.

    Parameters:

        A (numpy array): result of W.X (Weights.Inputs/features).

    Returns:

        (int): Returning 1.
   """

    # print("in softmax der")
    # temporaryly
    return 1


def relu(z):
    """ This function applies relu function(has mathematical form) as an activation function of a node for forwardpropagation.

    Parameters:

        z (numpy array): result of W.X (Weights.Inputs/features).

    Returns:

        (numpy array): Returning array of same size as input "z" after applying "max(0,input)".
   """

    A = np.maximum(0, z)
    return A


def relu_der(A):
    """ This function applies relu derivative function(has mathematical form = 1) for backpropagation

    Parameters:

        A (numpy array): result of W.X (Weights.Inputs/features)

    Returns:

        (numpy array): Returning array of same size as input "A", has 2 conditions; if less than zero then 0, else 1.
   """

    Z = np.array(A, copy=True)
    der = np.array(A, copy=True)

    der[Z <= 0] = 0
    der[Z > 0] = 1

    return der


def determine_der_act_func(func):
    """ This function works as a switch, returns the right dervative function for backpropagation as an opposite operation of applied activation function in forwardpropagation.

    Parameters:

        func (method): The activation function used in forwardpropagation.

    Returns:

        (method): Returning method of selective derivative activation function to make backpropagation
   """
    if func == sigmoid:
        return sigmoid_der
    elif func == tanh:
        return tanh_der
    elif func == relu:
        return relu_der
    elif func == softmax:
        return softmax_der
    elif func == identity:
        return identity_der



def cross_entropy(m, A, Y):  # Log LikelihoodLoss Function - Logistic Regression Sigmoid Activation Function
    """Log LikelihoodLoss Function - Logistic Regression Sigmoid Activation Function

    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        cost(float): the total loss
    """
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    return cost


def cross_entropy_der(m, A, Y):
    """The Derivative of Log LikelihoodLoss Function
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        (Array of floats): The derivative values of cost function
    """
    return ((-1 * Y) / A) + ((1 - Y) / (1 - A))


def perceptron_criteria(m, A, Y):
    """Perceptron Criteria

    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        cost(float): the total loss
    """


    cost = (1 / m) * np.sum(np.maximum(0, - Y * A))
    return cost


def perceptron_criteria_der(m, A, Y):
    """The Derivative of Perceptron Criteria loss Function
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        b(Array of floats): The derivative values of cost function
    """
    A.reshape(m)
    Y.reshape(m)
    p = Y * A
    b = np.zeros(A.shape)
    b[p > 0] = 0
    b[p <= 0] = -Y[p <= 0]
    b[b == 0] = -1
    return b  # .reshape(1,m)


def svm(m, A, Y):
    """Hinge Loss (Soft Margin) SVM

    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        cost(float): the total loss
    """
    cost = (1 / m) * np.sum(np.maximum(0, 1 - Y * A))
    return cost


def svm_der(m, A, Y):
    """The Derivative of Hinge Loss (Soft Margin) SVM Function
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        b(Array of floats): The derivative values of cost function
    """
    A.reshape(m)
    Y.reshape(m)
    p = Y * A - 1
    b = np.zeros(A.shape)
    b[p > 0] = 0
    b[p <= 0] = -Y[p <= 0]
    b[b == 0] = -1
    return b  # .reshape(1,m)





def cross_multi_class(m, A,
                      Y):  # Multiclass Log LikelihoodLoss Function - Logistic Regression SoftMax Activation Function
    # v1 = Y * A
    # v2 = np.max(v1,axis=0)
    # v3 = np.log(v2).sum()
    # return (-1 / m) * v3
    """Multiclass Log LikelihoodLoss Function - Logistic Regression SoftMax Activation Function
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        cost(float): the total loss
    """
    cost = (-1 / m) * np.sum((Y) * (np.log(A)))
    # print("in cross multi")
    return cost


def cross_multi_class_der(m, A, Y):
    """The Derivative of Multiclass Log LikelihoodLoss Function

    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label
    Returns:
        (Array of floats): The derivative values of cost function
    """
    z1 = np.array(A, copy=True)
    y1 = np.array(Y, copy=True)
    y1[y1 == 1] = -1
    return A - Y


def multiclass_perceptron_loss(m, A, Y):
    """Multiclass Perceptron Loss
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        cost(float): the total loss
    """
    D = np.maximum(A - np.max(Y * A, axis=1).reshape(m, 1), 0)
    cost = (1 / m) * np.sum(np.max(D, axis=1))
    return cost


def multiclass_perceptron_loss_der(m, A, Y):
    """The Derivative of Multiclass Perceptron Loss Function
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        p(Array of floats): The derivative values of cost function
    """
    # if np.arange(np.shape(A)) == np.argmax(Y*A):
    #    return - np.gradient(np.max(Y*A))
    # elif np.arange(np.shape(A)) != np.argmax(Y*A):
    #   return np.gradient(A)
    # else:
    #   return 0

    p = np.zeros(A.shape)
    p[np.arange(A.shape[0]) == np.argmax(Y * A)] = -np.max(Y * A)
    p[np.arange(A.shape[0]) != np.argmax(Y * A)] = np.gradient(A)[np.arange(A.shape[0]) != np.argmax(Y * A)]
    return p


def multiclass_svm(m, A, Y):
    """Multiclass Weston-Watkins SVM Loss
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label
    Returns:
        cost(float): the total loss
    """
    D = np.maximum(1 + A - np.max(Y * A, axis=1).reshape(m, 1), 0)
    cost = (1 / m) * (np.sum(np.sum(D, axis=1)) - m)
    return cost


def multiclass_svm_der(m, A, Y):
    """The Derivative of Multiclass Weston-Watkins SVM Loss Function
    Parameters:
        m (int):examples no.
        A (float vector): The output y_hat (score)
        Y (float vector): The label
    Returns:
        p (Array of floats): The derivative values of cost function
    """
    # if np.arange(np.shape(A)) == np.argmax(Y*A):
    #    return - np.gradient(np.max(Y*A))
    # elif np.arange(np.shape(A)) != np.argmax(Y*A):
    #    return np.gradient(A)
    # else:
    #    return 0

    p = np.zeros(A.shape)
    p[np.arange(A.shape[0]) == np.argmax(Y * A)] = -np.max(Y * A)
    p[np.arange(A.shape[0]) != np.argmax(Y * A)] = np.gradient(A)[np.arange(A.shape[0]) != np.argmax(Y * A)]
    return p


def multinomial_logistic_loss(m, A, Y):
    """Multinomial Logistic Regression using Softmax Activation
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        cost(float): the total loss
    """
    cost = np.sum(-np.max(A * Y) + np.log(np.sum(np.exp(A))))
    return cost


def multinomial_logistic_loss_der(m, A, Y):
    """The Derivative of Multinomial Logistic Regression using Softmax Activation Loss Function
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label
    Returns:
        (Array of floats): The derivative values of cost function
    """
    p = np.zeros(A.shape)
    p[Y == 1] = -(1 - A[Y == 1])
    p[Y == 0] = A[Y == 0]
    return p


def square_loss(m, A, Y):
    """Linear Regression Least squares using Identity Activation
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        cost(float): the total loss
    """
    # return (1/(2*m)) * np.sum(np.dot((A-Y).T,(A-Y)))
    cost = (1 / 2 * m) * np.sum(np.square(Y - A))
    return cost


def square_loss_der(m, A, Y):
    """The Derivative of Linear Regression Least squares using Identity Activation Loss Function
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
    (Array of floats): The derivative values of cost function
    """
    return A - Y


def logistic_sigmoid_loss(m, A, Y):
    """Logistic Regression using sigmoid Activation
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label
    Returns:
        cost(float): the total loss
    """
    cost = (-1 / m) * np.sum(np.log(0.5 * Y - 0.5 + A))
    return cost


def logistic_sigmoid_loss_der(m, A, Y):
    """The Derivative of Logistic Regression using sigmoid Activation Loss Function
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label
    Returns:
        (Array of floats): The derivative values of cost function
    """
    return (- 1) / (0.5 * Y - 0.5 + A)


def logistic_id_loss(m, A, Y):
    """Logistic Regression using Identity Activation
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        cost(float): the total loss
    """
    cost = (1 / m) * np.sum(np.log(1 + np.exp(- Y * A)))
    return cost


def logistic_id_loss_der(m, A, Y):
    """The Derivative of Logistic Regression using Identity Activation Loss Function
    Parameters:
        m(int):examples no.
        A(float vector): The output y_hat (score)
        Y(float vector): The label

    Returns:
        (Array of floats): The derivative values of cost function
    """
    return - (Y * np.exp(- Y * A)) / (1 + np.exp(- Y * A))


def determine_der_cost_func(func):
    """Determining The Derivative of The Loss function
    Parameters:
        func(string): The Loss function name

    Returns:
        (string): The Derivative of The Loss function
    """
    if func == cross_entropy:
        return cross_entropy_der
    if func == cross_multi_class:
        return cross_multi_class_der
    if func == square_loss:
        return square_loss_der
    if func == perceptron_criteria:
        return perceptron_criteria_der
    if func == svm:
        return svm_der
    if func == multiclass_perceptron_loss:
        return multiclass_perceptron_loss_der
    if func == multiclass_svm:
        return multiclass_svm_der
    if func == multinomial_logistic_loss:
        return multinomial_logistic_loss_der

def conf_mat (A,Y,thres=0.5):
    '''This function calculates the confusion matrix from predicted and truth matrices
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float): threshold value, values above it are considered true
    Returns:
        (numpy matrix): confusion matrix
    '''

    if Y.shape[0] == 1:
        predictions = (A > thres) * 1
        confusion_matrix =np.zeros((2,2))
        confusion_matrix[0][0] =float(np.dot(Y, predictions.T))
        confusion_matrix[0][1] =float(np.dot(1-Y, predictions.T))
        confusion_matrix[1][0] =float(np.dot(Y, 1-predictions.T))
        confusion_matrix[1][1] =float(np.dot(1-Y, 1-predictions.T))
    else:
        predectin = A.argmax(axis=0) #tuble containing predction of each example
        truth = Y.argmax(axis=0)     #tuble containing truth of each example
        confusion_matrix= np.zeros((Y.shape[0],Y.shape[0]))
        for i in range(0,Y.shape[1]):
            confusion_matrix[truth[i]][predectin [i]]+=1

    return confusion_matrix

def print_conf_mat(A,Y,thres=0.5):
    '''This function prints the confusion matrix from predicted and truth matrices
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float) : threshold value, values above it are considered true
    '''
    ax = sns.heatmap(conf_mat(A,Y), annot=True, linewidth=0.5)
    plt.show()

def conf_table(cnf_matrix):
    '''Tis function calculates the confusion table from confusion matrix
    Parameters:
        cnf_matrix (numpy matrix): confusion matrix
    Returns:
        (tuple of floats): representing TP for each class
        (tuple of floats): representing FP for each class
        (tuple of floats): representing TN for each class
        (tuple of floats): representing FN for each class
    '''
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    return FP,FN,TP,TN

def accuracy_score(A,Y,thres=0.5):
    '''This function calculates the accuracy of the model using predicted and truth values
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float): threshold value, values above it are considered true
    Returns:
        (float): accuracy
    '''
    if Y.shape[0] == 1:
        predictions = (A > thres) * 1
        ACC = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size))
    else:
        FP, FN, TP, TN = conf_table(conf_mat(A, Y))
        ACC = (TP + TN+0.00000000001) / (TP + FP + FN + TN+0.00000000001)
        ACC = np.sum(ACC) / ACC.shape[0]

    return ACC * 100

def precision_score(A, Y, thres=0.5):
    '''This function calculates the precision of the model using predicted and truth values
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float): threshold value, values above it are considered true
    Returns:
        (float): precision
    '''
    if Y.shape[0] == 1:
        predictions = (A > thres) * 1
        prec = float(np.dot(Y,predictions.T)/float(np.dot(Y,predictions.T)+np.dot(1-Y,predictions.T)))
    else:
        FP, FN, TP, TN = conf_table(conf_mat(A, Y))
        prec = (TP+0.00000000001)/(TP + FP+0.00000000001)
        prec = np.sum(prec) / prec.shape[0]
    return prec * 100


def recall_score(A,Y,thres=0.5):
    '''This function calculates the recall of the model using predicted and truth values
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float): threshold value, values above it are considered true
    Returns:
        (float): recall
    '''

    if Y.shape[0] == 1:
        predictions = (A > thres) * 1
        rec = float(np.dot(Y,predictions.T)/float(np.dot(Y,predictions.T)+np.dot(Y,1-predictions.T)))
    else:
        FP, FN, TP, TN = conf_table(conf_mat(A, Y))
        rec = (TP+0.00000000001) / (TP + FN+0.00000000001)
        rec = np.sum(rec) / rec.shape[0]

    return rec * 100

def f1_score(A,Y,thres=0.5):
    '''This function calculates the f1_score of the model using predicted and truth values
    Parameters:
        A (numpy matrix): predicted classification for each example
        Y (numpy matrix): true classification for each example
        thres (float) : threshold value, values above it are considered true
    Returns:
        (float): f1_score
    '''

    prec=precision_score(A, Y)
    rec=recall_score(A,Y)
    f1=float((2*prec*rec)/float(prec+rec))
    return f1

def gd_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1,
            reg_term=0, batch_size=0, param_dic=None, drop=0):
    """The function applies the  gradient descent optimizer to update the weight and bias parameters.

    Parameters:

        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: not used in this function.
        drop: dropout parameter to have the option of using the dropout technique.

    Returns:

        dictionary:parameters a dictionary that contains the updated weights and biases
        array:Costs an array that contain the cost of each iteration
    """

    costs = []

    if batch_size == 0:
        for i in range(0, num_iterations):

            Alast, cache = model.forward_propagation(X, drop)  # **

            cost = model.compute_cost(Alast, Y)
            if reg_term != 0:
                for key in model.parameters:
                    cost += (reg_term / X.shape[1]) * np.sum(model.parameters[key] ** 2)

            grads = model.backward_propagation(X, Y)

            parameters = model.update_parameters(grads, learning_rate=learning_rate, reg_term=reg_term, m=X.shape[1])

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

    else:
        for i in range(0, num_iterations):
            for j in range(int(X.shape[1] / batch_size)):

                Alast, cache = model.forward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size], drop)  # **

                cost = model.compute_cost(Alast, Y[:, j * batch_size:(j * batch_size) + batch_size])
                if reg_term != 0:
                    for key in model.parameters:
                        cost += (reg_term / X[:, j * batch_size:(j * batch_size) + batch_size].shape[1]) * np.sum(
                            model.parameters[key] ** 2)  # **

                grads = model.backward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size],
                                                   Y[:, j * batch_size:(j * batch_size) + batch_size])

                parameters = model.update_parameters(grads, learning_rate=learning_rate, reg_term=reg_term,
                                                     m=X[:, j * batch_size:(j * batch_size) + batch_size].shape[
                                                         1])  # **

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs


def adagrad_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1,
                 reg_term=0, batch_size=0, param_dic=None, drop=0):
    """The function applies the adagrad optimizer to update the weight and bias parameters.

    Parameters:

        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: not used in this function.
        drop: dropout parameter to have the option of using the dropout technique.

    Returns:

        dictionary: parameters a dictionary that contains the updated weights and biases
        array: Costs an array that contain the cost of each iteration
    """
    costs = []
    adagrads = {}

    if batch_size == 0:
        for i in range(0, num_iterations):

            Alast, cache = model.forward_propagation(X, drop)

            cost = model.compute_cost(Alast, Y)
            if reg_term != 0:
                for key in model.parameters:
                    cost += (reg_term / X.shape[1]) * np.sum(model.parameters[key] ** 2)

            grads = model.backward_propagation(X, Y)
            if i == 0:
                for key in grads:
                    adagrads[key] = np.square(grads[key])
            else:
                for key in grads:
                    adagrads[key] = adagrads[key] + np.square(grads[key])

            parameters = model.update_parameters_adagrad(grads, adagrads, learning_rate=learning_rate,
                                                         reg_term=reg_term, m=X.shape[1])

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

    else:
        for i in range(0, num_iterations):
            for j in range(int(X.shape[1] / batch_size)):

                Alast, cache = model.forward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size], drop)

                cost = model.compute_cost(Alast, Y[:, j * batch_size:(j * batch_size) + batch_size])
                if reg_term != 0:
                    for key in model.parameters:
                        cost += (reg_term / X[:, j * batch_size:(j * batch_size) + batch_size].shape[1]) * np.sum(
                            model.parameters[key] ** 2)

                grads = model.backward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size],
                                                   Y[:, j * batch_size:(j * batch_size) + batch_size])
                if i == 0:
                    for key in grads:
                        adagrads[key] = np.square(grads[key])
                else:
                    for key in grads:
                        adagrads[key] = adagrads[key] + np.square(grads[key])

                parameters = model.update_parameters_adagrad(grads, adagrads, learning_rate=learning_rate,
                                                             reg_term=reg_term,
                                                             m=X[:, j * batch_size:(j * batch_size) + batch_size].shape[
                                                                 1])

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs


def RMS_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1,
             reg_term=0, batch_size=0, param_dic=None, drop=0):
    """The function applies the RMS optimizer to update the weight and bias parameters.

    Parameters:

    model (multilayer): instance of the multilayer class contains the models parameters to be updated.
    X: the input feature vector.
    Y: the labels.
    num_iterations: number of epochs.
    print_cost: optional parameter to show the cost function.
    print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
    cont: not used in this function
    learning_rate: learning rate to be used in updating the parameters.
    reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
    batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
    param_dic: the dictionary that contains the value of the hyper parameter rho
    drop: dropout parameter to have the option of using the dropout technique.

    Returns:

    dictionary:parameters a dictionary that contains the updated weights and biases
    array:Costs an array that contain the cost of each iteration
    """

    costs = []
    rho = param_dic["rho"]
    eps = param_dic["eps"]
    rmsgrads = {}

    if batch_size == 0:
        for i in range(0, num_iterations):

            Alast, cache = model.forward_propagation(X, drop)

            cost = model.compute_cost(Alast, Y)
            if reg_term != 0:
                for key in model.parameters:
                    cost += (reg_term / X.shape[1]) * np.sum(model.parameters[key] ** 2)

            grads = model.backward_propagation(X, Y)
            if i == 0:
                for key in grads:
                    rmsgrads[key] = (1 - rho) * np.square(grads[key])
            else:
                for key in grads:
                    rmsgrads[key] = (rho) * rmsgrads[key] + (1 - rho) * np.square(grads[key])

            parameters = model.upadte_patameters_RMS(grads, rmsgrads, learning_rate=learning_rate, reg_term=reg_term,
                                                     m=X.shape[1], eps=eps)

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

    else:
        for i in range(0, num_iterations):
            for j in range(int(X.shape[1] / batch_size)):

                Alast, cache = model.forward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size], drop)

                cost = model.compute_cost(Alast, Y[:, j * batch_size:(j * batch_size) + batch_size])
                if reg_term != 0:
                    for key in model.parameters:
                        cost += (reg_term / X[:, j * batch_size:(j * batch_size) + batch_size].shape[1]) * np.sum(
                            model.parameters[key] ** 2)

                grads = model.backward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size],
                                                   Y[:, j * batch_size:(j * batch_size) + batch_size])
                if i == 0:
                    for key in grads:
                        rmsgrads[key] = np.square(grads[key])
                else:
                    for key in grads:
                        rmsgrads[key] = (rho) * rmsgrads[key] + (1 - rho) * np.square(grads[key])

                parameters = model.upadte_patameters_RMS(grads, rmsgrads, learning_rate=learning_rate,
                                                         reg_term=reg_term,
                                                         m=X[:, j * batch_size:(j * batch_size) + batch_size].shape[1],
                                                         eps=eps)

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs


def adadelta_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1,
                  reg_term=0, batch_size=0, param_dic=None, drop=0):
    """The function applies the adadelta optimizer to update the weight and bias parameters.

    Parameters:

        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: the dictionary that contains the value of the hyper parameters rho and epsilon.
        drop: dropout parameter to have the option of using the dropout technique.

    Returns:

        dictionary: parameters a dictionary that contains the updated weights and biases
        array: Costs an array that contain the cost of each iteration
    """

    costs = []
    rho = param_dic["rho"]
    eps = param_dic["eps"]
    adadeltagrads = {}
    segma = {}
    delta = {}

    if batch_size == 0:
        for i in range(0, num_iterations):

            Alast, cache = model.forward_propagation(X, drop)

            cost = model.compute_cost(Alast, Y)
            if reg_term != 0:
                for key in model.parameters:
                    cost += (reg_term / X.shape[1]) * np.sum(model.parameters[key] ** 2)

            grads = model.backward_propagation(X, Y)
            if i == 0:
                for key in grads:
                    adadeltagrads[key] = np.square(grads[key])
                    segma[key] = (np.random.randn(grads[key].shape[0], grads[key].shape[1]) + 2)
                    delta[key] = np.sqrt(segma[key] / (adadeltagrads[key]) + eps) * grads[key]
            else:
                for key in grads:
                    adadeltagrads[key] = adadeltagrads[key] + np.square(grads[key])
                    segma[key] = (rho) * segma[key] + (1 - rho) * np.square(delta[key])
                    delta[key] = np.sqrt(segma[key] / (adadeltagrads[key]) + eps) * grads[key]

            parameters = model.upadte_patameters_adadelta(grads, delta, learning_rate=learning_rate, reg_term=reg_term,
                                                          m=X.shape[1])

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

    else:
        for i in range(0, num_iterations):
            for j in range(int(X.shape[1] / batch_size)):

                Alast, cache = model.forward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size], drop)

                cost = model.compute_cost(Alast, Y[:, j * batch_size:(j * batch_size) + batch_size])
                if reg_term != 0:
                    for key in model.parameters:
                        cost += (reg_term / X[:, j * batch_size:(j * batch_size) + batch_size].shape[1]) * np.sum(
                            model.parameters[key] ** 2)

                grads = model.backward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size],
                                                   Y[:, j * batch_size:(j * batch_size) + batch_size])
                if i == 0:
                    for key in grads:
                        adadeltagrads[key] = np.square(grads[key])
                        segma[key] = (np.random.randn(grads[key].shape[0], grads[key].shape[1]) + 100) * 0.00001
                        delta[key] = np.sqrt(segma[key] / (adadeltagrads[key]) + eps) * grads[key]
                else:
                    for key in grads:
                        adadeltagrads[key] = adadeltagrads[key] + np.square(grads[key])
                        segma[key] = (rho) * segma[key] + (1 - rho) * np.square(delta[key])
                        delta[key] = np.sqrt(segma[key] / (adadeltagrads[key]) + eps) * grads[key]

                parameters = model.upadte_patameters_adadelta(grads, delta, learning_rate=learning_rate,
                                                              reg_term=reg_term, m=
                                                              X[:, j * batch_size:(j * batch_size) + batch_size].shape[
                                                                  1])

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs


def adam_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1,
              reg_term=0, batch_size=0, param_dic=None, drop=0):
    """The function applies the adam optimizer to update the weight and bias parameters.

    Parameters:

        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: the dictionary that contains the value of the hyper parameters rho , rhof and epsilon.
        drop: dropout parameter to have the option of using the dropout technique.

    Returns:

        dictionary:parameters a dictionary that contains the updated weights and biases
        array:Costs an array that contain the cost of each iteration
       """

    costs = []
    rho = param_dic["rho"]
    eps = param_dic["eps"]
    rhof = param_dic["rhof"]
    adamgrads = {}
    Fgrads = {}
    if batch_size == 0:
        for i in range(0, num_iterations):

            Alast, cache = model.forward_propagation(X, drop)

            cost = model.compute_cost(Alast, Y)
            if reg_term != 0:
                for key in model.parameters:
                    cost += (reg_term / X.shape[1]) * np.sum(model.parameters[key] ** 2)

            grads = model.backward_propagation(X, Y)
            if i == 0:
                for key in grads:
                    adamgrads[key] = (1 - rho) * np.square(grads[key])
                    Fgrads[key] = (1 - rhof) * grads[key]

            else:
                for key in grads:
                    adamgrads[key] = (rho) * adamgrads[key] + (1 - rho) * np.square(grads[key])
                    Fgrads[key] = (rho) * Fgrads[key] + (1 - rhof) * grads[key]
            alpha_t = learning_rate * np.sqrt((1 - rho ** (num_iterations)) / (1 - rhof ** (num_iterations)))

            parameters = model.update_parameters_adam(grads, adamgrads, Fgrads, learning_rate=alpha_t,
                                                      reg_term=reg_term, m=X.shape[1], eps=eps)

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

    else:
        for i in range(0, num_iterations):
            for j in range(int(X.shape[1] / batch_size)):

                Alast, cache = model.forward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size], drop)

                cost = model.compute_cost(Alast, Y[:, j * batch_size:(j * batch_size) + batch_size])
                if reg_term != 0:
                    for key in model.parameters:
                        cost += (reg_term / X[:, j * batch_size:(j * batch_size) + batch_size].shape[1]) * np.sum(
                            model.parameters[key] ** 2)

                grads = model.backward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size],
                                                   Y[:, j * batch_size:(j * batch_size) + batch_size])
                if i == 0:
                    for key in grads:
                        adamgrads[key] = (1 - rho) * np.square(grads[key])
                        Fgrads[key] = (1 - rhof) * grads[key]
                else:
                    for key in grads:
                        adamgrads[key] = (rho) * adamgrads[key] + (1 - rho) * np.square(grads[key])
                        Fgrads[key] = (rho) * Fgrads[key] + (1 - rhof) * grads[key]
                alpha_t = learning_rate * np.sqrt((1 - rho ** (num_iterations)) / (1 - rhof ** (num_iterations)))

                parameters = model.update_parameters_adam(grads, adamgrads, Fgrads, learning_rate=alpha_t,
                                                          reg_term=reg_term,
                                                          m=X[:, j * batch_size:(j * batch_size) + batch_size].shape[1],
                                                          eps=eps)

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs


def mom_optm(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1,
             reg_term=0, batch_size=0, param_dic=None, drop=0):
    """The function applies the momentum optimizer to update the weight and bias parameters.

    Parameters:

        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: the dictionary that contains the value of the hyper parameters beta.
        drop: dropout parameter to have the option of using the dropout technique.

    Returns:

        dictionary:parameters a dictionary that contains the updated weights and biases
        array:Costs an array that contain the cost of each iteration
           """

    costs = []

    beta = param_dic['beta']
    momen_grad = {}

    if batch_size == 0:
        for i in range(0, num_iterations):

            Alast, cache = model.forward_propagation(X, drop)
            cost = model.compute_cost(Alast, Y)
            if reg_term != 0:
                for key in model.parameters:
                    cost += (reg_term / X.shape[1]) * np.sum(model.parameters[key] ** 2)

            grads = model.backward_propagation(X, Y)

            if i == 0:
                for key in grads:
                    momen_grad[key] = (1 - beta) * grads[key]
            else:
                for key in grads:
                    momen_grad[key] = beta * momen_grad[key] + (1 - beta) * grads[key]

            parameters = model.update_parameters(momen_grad, learning_rate=learning_rate, reg_term=reg_term,
                                                 m=X.shape[1])

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

    else:
        for i in range(0, num_iterations):
            for j in range(int(X.shape[1] / batch_size)):

                Alast, cache = model.forward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size], drop)

                cost = model.compute_cost(Alast, Y[:, j * batch_size:(j * batch_size) + batch_size])
                if reg_term != 0:
                    for key in model.parameters:
                        cost += (reg_term / X[:, j * batch_size:(j * batch_size) + batch_size].shape[1]) * np.sum(
                            model.parameters[key] ** 2)

                grads = model.backward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size],
                                                   Y[:, j * batch_size:(j * batch_size) + batch_size])

                if i == 0:
                    for key in grads:
                        momen_grad[key] = (1 - beta) * grads[key]
                else:
                    for key in grads:
                        momen_grad[key] = beta * momen_grad[key] + (1 - beta) * grads[key]

                parameters = model.update_parameters(momen_grad, learning_rate=learning_rate, reg_term=reg_term,
                                                     m=X[:, j * batch_size:(j * batch_size) + batch_size].shape[1])

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs


def gd_optm_steepst(model, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0,
                    learning_rate=0.01, reg_term=0, batch_size=0, param_dic=None, drop=0):
    """The function applies the steepest gradient descent optimizer to update the weight and bias parameters.

    Parameters:

        model (multilayer): instance of the multilayer class contains the models parameters to be updated.
        X: the input feature vector.
        Y: the labels.
        num_iterations: number of epochs.
        print_cost: optional parameter to show the cost function.
        print_cost_each: this parameter is used when "print_cost" is set True to specify when to print the cost ie: after how many iterations.
        cont: not used in this function
        learning_rate: learning rate to be used in updating the parameters.
        reg_term: lamda term added to the loss function to prevent over fitting. This parameter can be set to zero if no regulization is needed.
        batch_size: This parameter is used to specify if the learning process is batch , online or minibatch.
        param_dic: not used in this function.
        drop: dropout parameter to have the option of using the dropout technique.

    Returns:

        dictionary:parameters a dictionary that contains the updated weights and biases
        array:Costs an array that contain the cost of each iteration
           """
    costs = []

    if batch_size == 0:
        for i in range(0, num_iterations):

            Alast, cache = model.forward_propagation(X, drop)  # **

            cost = model.compute_cost(Alast, Y)
            if reg_term != 0:
                for key in model.parameters:
                    cost += (reg_term / X.shape[1]) * np.sum(model.parameters[key] ** 2)

            grads = model.backward_propagation(X, Y)
            m = Alast.shape[1]
            learning_rate = 100 * np.amin((- 1 / m) * (
            Y * np.log(np.abs(Alast - (learning_rate * model.cost_func_der(m, Alast, Y)))) + (1 - Y) * (
            np.log(np.abs(1 - (Alast - learning_rate * model.cost_func_der(m, Alast, Y)))))))

            parameters = model.update_parameters(grads, learning_rate=learning_rate, reg_term=reg_term, m=X.shape[1])

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

    else:
        for i in range(0, num_iterations):
            for j in range(int(X.shape[1] / batch_size)):

                Alast, cache = model.forward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size], drop)  # **

                cost = model.compute_cost(Alast, Y[:, j * batch_size:(j * batch_size) + batch_size])
                if reg_term != 0:
                    for key in model.parameters:
                        cost += (reg_term / X[:, j * batch_size:(j * batch_size) + batch_size].shape[1]) * np.sum(
                            model.parameters[key] ** 2)  # **
                grads = model.backward_propagation(X[:, j * batch_size:(j * batch_size) + batch_size],
                                                   Y[:, j * batch_size:(j * batch_size) + batch_size])
                m = Alast.shape[1]
                learning_rate = 100 * np.amin((- 1 / m) * (
                Y * np.log(np.abs(Alast - (learning_rate * model.cost_func_der(m, Alast, Y)))) + (1 - Y) * (
                np.log(np.abs(1 - (Alast - learning_rate * model.cost_func_der(m, Alast, Y)))))))

                parameters = model.update_parameters(grads, learning_rate=learning_rate, reg_term=reg_term,
                                                     m=X[:, j * batch_size:(j * batch_size) + batch_size].shape[
                                                         1])  # **

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

class MultiLayer:
    def __init__(self, number_of_neurons=0, cost_func=cross_entropy):
        """ init of the class multilayer and needed variables
        variables:
            w,b lists for weights
            parameters dic for weights in the form of parameters['W1']
            layers_size for size of each layer
            number_of_input_neurons
            act_func list for activations of each layer
            derivative_act_func list for backward activations derivative functions
            cost_func the choosen cost functions
        parmeters:
            (method) : the cost function of model
        returns:
            (None)
        """
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
        """ add the input layer of the model
        parmeters:
            size (int) : size of input layer
        retruns:
            (None)
        """
        self.number_of_input_neurons = size
        self.layer_size.append(size)

    def addHidenLayer(self, size, act_func=sigmoid):
        """ add a hidden layer of the model
        parmeters:
            size (int) : size of input layer
            act_func (function) : the activation function of the layer

        retruns:
            (None)
        """
        self.layer_size.append(size)
        self.act_func.append(act_func)
        self.derivative_act_func.append(determine_der_act_func(act_func))

    def addOutputLayer(self, size, act_func=sigmoid):
        """ add the output layer of the model
        parmeters:
            size (int) : size of input layer
            act_func (function) : the activation function of the layer

        retruns:
            (None)
        """
        self.number_of_outputs = size
        self.layer_size.append(size)
        self.act_func.append(act_func)
        self.derivative_act_func.append(determine_der_act_func(act_func))

    def initialize_parameters(self, seed=2):  # ,init_func=random_init_zero_bias):
        """ initialize_weights of the model at the start with xavier init
        parmeters:
            seed (int) : seed for random function
        retruns:
            paramters
        """

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

    def forward_propagation(self, X, drop=0):
        """ forward propagation through the layers
        parmeters:
            X (np.array) : input feature vector
            drop (float) : propablity to keep neurons or shut down

        retruns:
            cashe (dic) : the output of each layer in the form of cashe['Z1']
            Alast (np.array) : last layer activations
        """

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
        """ cahnge the initial cost function
        parmeters:
            cost_funct (function) : the new function

        retruns:
            cashe (dic) : the output of each layer in the form of cashe['Z1']
            Alast (np.array) : last layer activations
        """
        self.cost_func = cost_func
        self.cost_func_der = determine_der_cost_func(cost_func)

    def compute_cost(self, Alast, Y):
        """ compute cost of the given examples
        parmeters:
            Alast (np.array) : model predictions
            Y (np.array) : True labels

        retruns:
            cost (float) : cost output
        """
        m = Alast.shape[1]
        return self.cost_func(m, Alast, Y)

    def backward_propagation(self, X, Y):
        """ compute cost of the given examples
        parmeters:
            Alast (np.array) : model predictions
            Y (np.array) : True labels

        retruns:
            grads (dic) : all gridients of wieghts and biasses
        """

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
        """ set an external cache
        parmeters:
            X (np.array) : input feature vector
            cache (dic) :  output of each layer

        retruns:
            (None)
        """
        self.cache = cache
        self.prev = []
        self.prev.append((1, X))
        for i in range(int(len(cache.keys()) / 2)):
            A, Z = cache["A" + str(i + 1)], cache["Z" + str(i + 1)]
            self.prev.append((Z, A))

    def set_parameters(self, para):
        """ set an external parmeters
        parmeters:
            para (dic) :  the weights and biasses

        retruns:
            (None)
        """
        self.parameters = para
        self.w = []
        self.b = []
        for i in range(int(len(para.keys()) / 2)):
            W, b = para["W" + str(i + 1)], para["b" + str(i + 1)]
            self.w.append(W)
            self.b.append(b)

    def set_parameters_internal(self):
        """ set an internal parmeters this is used by model during training
        parmeters:
            (None)

        retruns:
            (None)
        """
        self.parameters = {}
        for i in range(len(self.w)):
            self.parameters["W" + str(i + 1)] = self.w[i]
            self.parameters["b" + str(i + 1)] = self.b[i]

    def update_parameters(self, grads, learning_rate=1.2, reg_term=0, m=1):
        """ update parameters using grads
        parmeters:
            grads (dic) :  the gradient of weights and biases
            learning_rate (float) : the learn rate hyper parameter
            reg_term (float) : the learn rate hyper parameter

        returns:
            dictionary contains the updated parameters
        """

        for i in range(len(self.w)):
            self.w[i] = (1 - reg_term / m) * self.w[i] - learning_rate * grads["dW" + str(i + 1)]
            self.b[i] = (1 - reg_term / m) * self.b[i] - learning_rate * grads["db" + str(i + 1)]

        self.set_parameters_internal()

        return self.parameters

    def update_parameters_adagrad(self, grads, adagrads, learning_rate=1.2, reg_term=0, m=1):
        """ update parameters using adagrad
        parameters:
            grads (dic) :  the gradient of weights and biases
            adagrads(dic): the square of the gradiant
            learning_rate (float) : the learn rate hyper parameter
            reg_term (float) : the learn rate hyper parameter

        returns:
            dictionary contains the updated parameters
        """

        for i in range(len(self.w)):
            self.w[i] = (1 - reg_term / m) * self.w[i] - (learning_rate / (
            np.sqrt(adagrads["dW" + str(i + 1)]) + 0.000000001)) * grads["dW" + str(i + 1)]
            self.b[i] = (1 - reg_term / m) * self.b[i] - (learning_rate / (
            np.sqrt(adagrads["db" + str(i + 1)]) + 0.000000001)) * grads["db" + str(i + 1)]
        self.set_parameters_internal()

        return self.parameters

    def upadte_patameters_RMS(self, grads, rmsgrads, learning_rate=1.2, reg_term=0, m=1, eps=None):
        """ update parameters using RMS gradient
        parameters:
            grads (dic) :  the gradient of weights and biases
            rmsgrads(dic): taking rho multiplied by the square of previous grads and (1-rho) multiplied by the square of current grads
            learning_rate (float) : the learn rate hyper parameter
            reg_term (float) : the learn rate hyper parameter
            eps(float) : the small value added to rmsgrads to make sure there is no division by zero

        returns:
            dictionary contains the updated parameters
        """

        for i in range(len(self.w)):
            self.w[i] = (1 - reg_term / m) * self.w[i] - (
                                                         learning_rate / (np.sqrt(rmsgrads["dW" + str(i + 1)]) + eps)) * \
                                                         grads["dW" + str(i + 1)]
            self.b[i] = (1 - reg_term / m) * self.b[i] - (
                                                         learning_rate / (np.sqrt(rmsgrads["db" + str(i + 1)]) + eps)) * \
                                                         grads["db" + str(i + 1)]
        self.set_parameters_internal()

        return self.parameters

    def upadte_patameters_adadelta(self, grads, delta, learning_rate=1.2, reg_term=0, m=1):
        """ update parameters using RMS gradient
        parameters:
            grads (dic) :  the gradient of weights and biases, note: this parameter is not used in this function
            delta(dic): dictionary contains the values that should be subtracted from current parameters to be updated
            learning_rate (float) : the learn rate hyper parameter , note: this parameter is not used in this function
            reg_term (float) : the learn rate hyper parameter

        returns:
            dictionary contains the updated parameters
        """

        for i in range(len(self.w)):
            self.w[i] = (1 - reg_term / m) * self.w[i] - delta["dW" + str(i + 1)]
            self.b[i] = (1 - reg_term / m) * self.b[i] - delta["db" + str(i + 1)]
        self.set_parameters_internal()

        return self.parameters

    def update_parameters_adam(self, grads, adamgrads, Fgrads, learning_rate=1.2, reg_term=0, m=1, eps=None):
        """ update parameters using RMS gradient
        parameters:
            grads (dic) :  the gradient of weights and biases , note: grads is not used in this function
            adamgrads(dic): taking rho multiplied by the square of previous grads and (1-rho) multiplied by the square of current grads
            Fgrads(dic): taking rhof multiplied by the  previous grads and (1-rhof) multiplied by the  current grads
            learning_rate (float) : the learn rate hyper parameter (alpha_t not alpha)
            reg_term (float) : the learn rate hyper parameter
            eps(float) : the small value added to adamgrads to make sure there is no division by zero

        returns:
            dictionary contains the updated parameters
        """

        for i in range(len(self.w)):
            self.w[i] = (1 - reg_term / m) * self.w[i] - (learning_rate / np.sqrt(adamgrads["dW" + str(i + 1)] + eps)) * \
                                                         Fgrads["dW" + str(i + 1)]
            self.b[i] = (1 - reg_term / m) * self.b[i] - (learning_rate / np.sqrt(adamgrads["db" + str(i + 1)] + eps)) * \
                                                         Fgrads["db" + str(i + 1)]
        self.set_parameters_internal()

        return self.parameters

    def train(self, X, Y, num_iterations=10000, print_cost=False, print_cost_each=100, cont=0, learning_rate=1,
              reg_term=0, batch_size=0, opt_func=gd_optm, param_dic=None, drop=0):
        """ train giving the data and hpyerparmeters and optmizer type

        parmeters:
            X (np.array) : input feature vector
            Y (np.array) :  the true label
            num_of iterations (int) : how many epochs
            print cost (bool) : to print cost or not
            print cost_each (int) : to print cost each how many iterations
            learning_rate (float) : the learn rate hyper parmeter
            reg_term (float) : the learn rate hyper parmeter
            batch_size (int) : how big is the mini batch and 0 for batch gradint
            optm_func (function) : a function for calling the wanted optmizer

        retruns:
            parmeters (dic) : weights and biasses after training
            cost (float) : cost
        """

        parameters, costs = opt_func(self, X, Y, num_iterations, print_cost, print_cost_each, cont, learning_rate,
                                     reg_term, batch_size, param_dic, drop)
        return parameters, costs

    def predict(self, X):
        """ perdict classes or output
        parmeters:
            X (np.array) :  input feature vector

        retruns:
            Alast (np.array) : output of last layer
        """

        Alast, cache = self.forward_propagation(X)
        # predictions = (Alast > thres) * 1

        return Alast

    def test(self, X, Y, eval_func=accuracy_score):
        """ evalute model
        parmeters:
            X (np.array) :  input feature vector
            Y (np.array) :  the true label
            eval_func (function) : the method of evalution

        retruns:
            Alast (np.array) : output of last layer
        """

        Alast, cache = self.forward_propagation(X)

        acc = eval_func(Alast, Y)

        return acc
