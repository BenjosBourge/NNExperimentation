import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score


def activation_sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def get_outputs(W, b, X):
    # Z.shape = n x number of neuron
    Z = X.dot(W) + b # dot make a*w1, a*w2, a*w3... for all pair of inputs
    #this return a matrice of all the outputs, for all set of inputs
    return activation_sigmoid(Z)


def error(A, l):
    # when in the output layer, (y - A)
    # when in hidden layer, (w0 * l0 + w1 * l1 ...) dot product surely
    return A * (1 - A) * l


def forward_propagation(X0, W, b):
    A = []
    current_input = X0
    for i in range(len(W)):
        A.append(get_outputs(W[i], b[i], current_input))  # n x lenWi
        current_input = A[i]
    return A


def back_propagation(X0, Abis, W, b, learning_rate, y):
    A = Abis.copy()
    A.insert(0, X0)

    # the goal is to have A containing n neurons
    # and W containing n - 1 weights

    L = (y - A[-1]) # last iteration
    for i in reversed(range(len(W))):
        E = error(A[i + 1], L)  # n x nb neurons this layer
        W[i] = W[i] + learning_rate * A[i].T.dot(E)  # update the weights
        b[i] = b[i] + learning_rate * E.mean()
        L = E.dot(W[i].T)  # n x nb neurons this layer
    return W, b


def train_nn(X0, y, learning_rate = 0.2, n_iter = 1000):
    # X0 are all the inputs n x nb_inputs

    # there is no weights and bias for layer 0, because it is the layer of the inputs

    layers = [6, 6, 1]
    layers.insert(0, X0.shape[1])

    W = []
    b = []

    L = []

    for i in range(len(layers) - 1):
        W.append(np.random.randn(layers[i], layers[i + 1])) # first one, is the len of previous layer, second one is the len of layer
        b.append(np.random.randn(layers[i + 1]))


    for i in range(n_iter):
        #the matrixes allows only one matrix calculus for one iteration
        #forward propagation
        A = forward_propagation(X0, W, b)

        L.append((y - A[-1]).mean())

        #back propagation
        W, b = back_propagation(X0, A, W, b, learning_rate, y)

    plt.plot(L)
    plt.show()

    return W, b


def main():
    X, y = make_moons(n_samples=100, noise = 0.1)
    y = y.reshape((y.shape[0], 1)) #from a big array to a multiples little arrays

    W, b = train_nn(X, y)


    #this code only show the result
    plt.scatter(X[:,0], X[:,1], c=y, cmap='spring')
    plt.show()

    # Show the gradient in the outputs matplotlib
    show_gradient = True

    if show_gradient:
        min_xvalue = X[:, 0].min()
        min_yvalue = X[:, 1].min()
        max_xvalue = X[:, 0].max()
        max_yvalue = X[:, 1].max()
        gradient_values = np.zeros((10000, 2))
        for i in range(100):
            for j in range(100):
                gradient_values[i * 100 + j][0] = (i / 100) * (max_xvalue - min_xvalue) + min_xvalue
                gradient_values[i * 100 + j][1] = (j / 100) * (max_yvalue - min_yvalue) + min_yvalue

        gy = forward_propagation(gradient_values, W, b)[-1]
        plt.scatter(gradient_values[:, 0], gradient_values[:, 1], c=gy, cmap='RdBu')

    yy = forward_propagation(X, W, b)[-1] > 0.5
    plt.scatter(X[:, 0], X[:, 1], c=yy, cmap='spring')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
