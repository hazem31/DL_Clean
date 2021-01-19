import numpy as np


def gd_optm(model, X, Y, num_iterations=10000, print_cost=False ,print_cost_each=100, cont=0, learning_rate=1, regu_term=0, batch_size=0 , param_dic=None):

    costs = []

    if batch_size == 0:
        for i in range(0, num_iterations):

            Alast, cache = model.forward_propagation(X)

            cost = model.compute_cost(Alast, Y)

            grads = model.backward_propagation(X, Y)

            parameters = model.update_parameters(grads, learning_rate=learning_rate)

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

    else:
        for i in range(0, num_iterations):
            for j in range(int(X.shape[1]/batch_size)):

                Alast, cache = model.forward_propagation(X[:,j*batch_size:(j*batch_size)+batch_size])

                cost = model.compute_cost(Alast, Y[:,j*batch_size:(j*batch_size)+batch_size])

                grads = model.backward_propagation(X[:,j*batch_size:(j*batch_size)+batch_size], Y[:,j*batch_size:(j*batch_size)+batch_size])

                parameters = model.update_parameters(grads, learning_rate=learning_rate)

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs


def mom_optm(model, X, Y, num_iterations=10000, print_cost=False ,print_cost_each=100, cont=0, learning_rate=1, regu_term=0, batch_size=0 , param_dic=None):

    costs = []

    beta = param_dic['beta']
    momen_grad = {}

    if batch_size == 0:
        for i in range(0, num_iterations):

            Alast, cache = model.forward_propagation(X)

            cost = model.compute_cost(Alast, Y)

            grads = model.backward_propagation(X, Y)

            if i == 0:
                for key in grads:
                    momen_grad[key] = (1 - beta) * grads[key]
            else:
                for key in grads:
                    momen_grad[key] = beta * momen_grad[key] + (1 - beta) * grads[key]

            parameters = model.update_parameters(momen_grad, learning_rate=learning_rate)

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs

    else:
        for i in range(0, num_iterations):
            for j in range(int(X.shape[1]/batch_size)):

                Alast, cache = model.forward_propagation(X[:,j*batch_size:(j*batch_size)+batch_size])

                cost = model.compute_cost(Alast, Y[:,j*batch_size:(j*batch_size)+batch_size])

                grads = model.backward_propagation(X[:,j*batch_size:(j*batch_size)+batch_size], Y[:,j*batch_size:(j*batch_size)+batch_size])

                if i == 0:
                    for key in grads:
                        momen_grad[key] = (1 - beta) * grads[key]
                else:
                    for key in grads:
                        momen_grad[key] = beta * momen_grad[key] + (1 - beta) * grads[key]

                parameters = model.update_parameters(momen_grad, learning_rate=learning_rate)

            if print_cost and i % print_cost_each == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return parameters, costs
