import numpy as np


def batch_gd_optm(model, X, Y, num_iterations=10000, print_cost=False ,print_cost_each=100, cont=0, learning_rate=1, regu_term=0, batch_size=0):

    costs = []

    for i in range(0, num_iterations):

        Alast, cache = model.forward_propagation(X)

        cost = model.compute_cost(Alast, Y)

        grads = model.backward_propagation(X, Y)

        parameters = model.update_parameters(grads, learning_rate=learning_rate)

        if print_cost and i % print_cost_each == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters , costs