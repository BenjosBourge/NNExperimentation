import numpy as np
import pygame
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

    for i in range(len(layers) - 1):
        W.append(np.random.randn(layers[i], layers[i + 1])) # first one, is the len of previous layer, second one is the len of layer
        b.append(np.random.randn(layers[i + 1]))


    for i in range(n_iter):
        #the matrixes allows only one matrix calculus for one iteration
        #forward propagation
        A = forward_propagation(X0, W, b)

        #back propagation
        W, b = back_propagation(X0, A, W, b, learning_rate, y)

    return W, b


def color_by_coordinates(v):
    # Example pattern: gradient based on position
    v = np.clip(v, -1, 1)
    red = (np.clip(v, 0, 1) * 255)
    green = (-np.clip(v, -1, 0) * 255)
    blue = 128 * np.fmax(green / 255, red / 255)
    return 255 - red, 255 - green, 255 - blue


def main():
    X, y = make_moons(n_samples=100, noise = 0.1)
    y = y.reshape((y.shape[0], 1)) #from a big array to a multiples little arrays

    W, b = train_nn(X, y)

    yy = forward_propagation(X, W, b)[-1] > 0.5

    pygame.init()

    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pixel Array Display")

    square_size = 200
    square_x, square_y = (width - square_size) // 2, (height - square_size) // 2

    square_surface = pygame.Surface((square_size, square_size))

    pixels = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for x in range(square_size):
            for y in range(square_size):
                pixels[x, y] = color_by_coordinates((x / square_size) * 2 - 1)

        pygame.surfarray.blit_array(square_surface, pixels)
        screen.fill((30, 30, 30))
        screen.blit(square_surface, (square_x, square_y))
        pygame.display.flip()

    pygame.quit()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
