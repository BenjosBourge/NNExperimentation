import numpy as np
import pygame
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score

import random

from nn import NeuralNet

class Individual(NeuralNet):
    def __init__(self, layers):
        NeuralNet.__init__(self, layers)
        self.chromosome = self.set_random_chromosome(self.get_length_chromosome(layers))
        self.fitness = float('inf')

    def set_random_chromosome(self, length):
        chromosome = []
        for i in range(length):
            chromosome.append(np.random.uniform(-1, 1))
        return chromosome

    def get_length_chromosome(self, layers):
        l = 0
        for i in range(len(layers)-1):
            l += layers[i] * layers[i+1]
            l += layers[i+1]
        return l


class GeneticAlgorithms:
    def __init__(self, X, y, layers, population_size=100):
        self._population = []
        for i in range(population_size):
            self._population.append(Individual(layers))
            self._population[i].setup_training(X, y)
        self._X = X
        self._y = y

        self._bestIndividual = self._population[0]
        self._bestScore = float('inf')

    def choose_parents(self):
        for i in range(len(self._population)):
            yy = self._population[i].forward_propagation(self._X)[-1]
            fit = self._population[i].MSE(self._y, yy)
            self._population[i].fitness = fit
        np.random.shuffle(self._population)
        parents = []
        for i in range(int(len(self._population) / 4)):
            biggest = self._population[i * 4]
            for j in range(4):
                if self._population[i * 4 + j].fitness < biggest.fitness:
                    biggest = self._population[i * 4 + j]
            parents.append(biggest)
        return parents

    def mate(self, parents):
        newborns = []
        for i in range(int(len(parents) / 2)):
            new_chromosome = []
            for j in range(len(parents[i * 2].chromosome)):
                r = np.random.randint(0, 2)
                if r == 0:
                    new_chromosome.append(parents[i * 2].chromosome[j])
                else:
                    new_chromosome.append(parents[i * 2 + 1].chromosome[j])
            newborns.append(new_chromosome)
        return newborns

    def mutate(self):
        pass

    def kill(self, newborns):
        # replace killed by newborns
        np.random.shuffle(self._population)
        killed = []
        for i in range(int(len(self._population) / 8)):
            lowest = i * 8
            lowestValue = float('inf')
            for j in range(8):
                if self._population[i * 8 + j].fitness < lowestValue:
                    lowest = i * 8 + j
            killed.append(lowest)

        for i in range(len(killed)):
            self._population[killed[i]].set_wb_from_1D(newborns[i])


    def iterate(self, nb_iter=100):
        for i in range(nb_iter):
            parents = self.choose_parents()
            newborns = self.mate(parents)
            self.mutate()
            self.kill(newborns)

    def get_bestIndividualResult(self, X):
        return self._bestIndividual.forward_propagation(X)