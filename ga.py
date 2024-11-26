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
        self.killed = False
        self.gifted = False

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
    def __init__(self, X, y, layers, population_size=100, mode=0, nb_parthenogenesis=0, store_values=False):
        self._population = []
        for i in range(population_size):
            self._population.append(Individual(layers))
            self._population[i].setup_training(X, y)
        self._X = X
        self._y = y
        self._layers = layers
        self._death_mode = mode
        self._nb_parthenogenesis = nb_parthenogenesis

        self._bestIndividual = Individual(layers)
        self._bestChromosome = self._population[0].get_wb_as_1D()
        self._bestScore = float('inf')

        self._bestScoreThisIt = float('inf')
        self._meanScoreThisIt = float('inf')

        self._store_values = store_values
        self._means = []
        self._bests = []

    def evaluatePopulation(self):
        mean = 0
        self._bestScoreThisIt = float('inf')
        for i in range(len(self._population)):
            self._population[i].gifted = False
            yy = self._population[i].forward_propagation(self._X)[-1]
            fit = self._population[i].MSE(self._y, yy)
            self._population[i].fitness = fit
            mean += fit
            if fit < self._bestScore:
                self._bestChromosome = self._population[i].get_wb_as_1D()
                self._bestScore = fit
            if fit < self._bestScoreThisIt:
                self._bestScoreThisIt = fit
        mean /= len(self._population)
        self._meanScoreThisIt = mean

    # ------- kill and gift selection ------

    def worst_to_die(self, to_kill):
        killed = []
        for i in range(to_kill):
            lowest = 0
            lowestValue = float('inf')
            for j in range(len(self._population)):
                if self._population[j].killed:
                    continue
                if self._population[j].fitness < lowestValue:
                    lowest = j
                    lowestValue = self._population[j].fitness
            self._population[lowest].killed = True
            killed.append(lowest)
        return killed

    def tournament_to_die(self, to_kill):
        killed = []
        np.random.shuffle(self._population)
        nb_per_batch = int(len(self._population) / (to_kill))
        for i in range(to_kill):
            lowest = i * nb_per_batch
            for j in range(nb_per_batch):
                if self._population[i * nb_per_batch + j].fitness < self._population[lowest].fitness:
                    lowest = i * nb_per_batch + j
            killed.append(lowest)
        return killed

    def best_to_gift(self, to_gift):
        gifted = []
        for i in range(to_gift):
            highest = 0
            highestValue = float('-inf')
            for j in range(len(self._population)):
                if self._population[j].killed or self._population[j].gifted:
                    continue
                if self._population[j].fitness > highestValue:
                    highest = j
                    highestValue = self._population[j].fitness
            self._population[highest].gifted = True
            gifted.append(highest)
        return gifted

    # --------------------------------------

    def kill_and_gift(self, to_kill, to_gift):
        np.random.shuffle(self._population) # allow random
        killed = []
        if self._death_mode == 0:
            killed = self.worst_to_die(to_kill)
        else:
            killed = self.tournament_to_die(to_kill)
        gifted =  self.best_to_gift(to_gift)

        return killed, gifted

    def choose_parents(self, nb_killed):
        np.random.shuffle(self._population)
        parents = []
        nb_per_batch = int(len(self._population) / (nb_killed * 2))
        for i in range(nb_killed * 2):
            biggest = self._population[i * nb_per_batch]
            for j in range(nb_per_batch):
                if self._population[i * nb_per_batch + j].fitness > biggest.fitness:
                    biggest = self._population[i * nb_per_batch + j]
            parents.append(biggest)
        return parents

    def mate(self, parents, killed):
        newborns = []
        for i in range(len(killed)):
            new_chromosome = []
            for j in range(len(parents[i * 2].chromosome)):
                r = np.random.randint(0, 2)
                if r == 0:
                    new_chromosome.append(parents[i * 2].chromosome[j])
                else:
                    new_chromosome.append(parents[i * 2 + 1].chromosome[j])
            newborns.append(new_chromosome)

        for i in range(len(killed)):
            self._population[killed[i]].killed = False
            self._population[killed[i]].set_wb_from_1D(newborns[i])

    def parthenogenesis(self, nb, gifted, killed):
        for i in range(nb):
            self._population[killed[i]].set_wb_from_1D(self._population[gifted[i]].get_wb_as_1D())
            self._population[killed[i]].killed = False

    def mutate(self):
        nb_mutation = 80
        mutation_strength = 0.2
        for i in range(len(self._population)):
            if self._population[i].killed:
                continue
            for j in range(nb_mutation):
                random_index = random.randint(0, len(self._population[i].chromosome) - 1)
                self._population[i].chromosome[random_index] += random.uniform(-1, 1) * mutation_strength

    def iterate(self, nb_iter=100):
        for i in range(nb_iter):
            self.evaluatePopulation()
            killed, gifted = self.kill_and_gift(15, self._nb_parthenogenesis)
            self.parthenogenesis(self._nb_parthenogenesis, gifted, killed)
            del killed[:self._nb_parthenogenesis]
            self.mutate()
            parents = self.choose_parents(len(killed))
            self.mate(parents, killed)
            if self._store_values:
                self._means.append(self._meanScoreThisIt)
                self._bests.append(self._bestScoreThisIt)
        self._bestIndividual.set_wb_from_1D(self._bestChromosome)

    def get_bestIndividualResult(self, X):
        return self._bestIndividual.forward_propagation(X)

    def get_bestScore(self):
        return self._bestScore

    def get_bestScoreThisIt(self):
        return self._bestScoreThisIt

    def get_meanScoreThisIt(self):
        return self._meanScoreThisIt

    def getAllMeans(self):
        return self._means

    def getAllBests(self):
        return self._bests
