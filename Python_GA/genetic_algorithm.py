from game_objects import Snake
from neural_network import *
import pygame as pg
import numpy as np
import sys
import random
import pickle
import logging
import time
import os

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class Population:
    def __init__(self, population_size, game):
        self.game = game
        self.individuals = [Snake(game, None) for i in range(population_size)]
        self.current_individual_index = 0
        self.INDIVIDUALS_TO_DRAW = population_size
        self.population_size = population_size
        self.is_done = False

    def get_snake(self, index):
        return self.individuals[index]

    def update(self):
        self.all_done = True
        for ind in self.individuals:
            if ind.check_if_done() == False:
                ind.update()
                self.all_done = False
                continue

    def draw(self):
        for i in range(self.INDIVIDUALS_TO_DRAW):
            self.individuals[i].draw(255 - (i * (255/self.INDIVIDUALS_TO_DRAW)))

    def __str__(self):
        the_string = ''
        for i in range(len(self.individuals)):
            avr_steps_formatted = "{:,.2f}".format(self.individuals[i].average_steps).replace(',',"'")
            fitness_calculation_string = f"({self.individuals[i].record} * 5000) - ({self.individuals[i].deaths} * 150) - ({avr_steps_formatted} * 100) - ({self.individuals[i].penalties} * 1000)"
            formatted_fitness = "{:,.2f}".format(self.individuals[i].fitness)#.replace(',',"'")
            the_string += f"Individual {i}:\t Fitness= {BLUE}{formatted_fitness}{RESET}\t[{fitness_calculation_string}]\n"
        return the_string

class Genetic_algorithm:
    def __init__(self, number_of_generations, game):
        self.NUMBER_OF_GENERATIONS = number_of_generations
        self.current_generation = 0
        self.timestamp = time.strftime("%Y%m%d%H%M%S")
        self.game = game
        self.best_individual = None
        
        self.file_name = os.path.join("logs", f"results_V{self.timestamp}.log")
        logging.basicConfig(filename=self.file_name, filemode='w')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        print(RED + "GENERATION " + str(self.current_generation) + "... " + RESET, end='', flush=True)

    def update_best_individual(self, population):
        if self.best_individual == None:
            self.best_individual = population.individuals[-1] # Last element is the highest fitness individual
            return
        if self.best_individual.fitness < population.individuals[-1].fitness:
            self.best_individual = population.individuals[-1]

    def save_best_individual(self):
        if self.current_generation == 0:
            return
        folder_path = "saved_individuals"
        os.makedirs(folder_path, exist_ok=True)
        
        file_name = os.path.join(folder_path, f"best_individual_V{self.timestamp}.pkl")
        with open(file_name, 'wb') as file:
            pickle.dump(self.best_individual.brain, file)
            print(f"Best individual saved to {file_name}")

    def next_generation(self, population):
        print(GREEN + " DONE" + RESET)

        #Logging
        self.logger.debug(f"\nGENERATION {str(self.current_generation)}...")

        # These two have to be sorted the same way, so that the indices match
        fitness_list = self.calculate_population_fitness(population)
        self.sort_population_on_fitness(population)
        fitness_list.sort()

        self.update_best_individual(population)

        # Debugging
        print(population)
        
        #Logging
        for i in range(len(population.individuals)):
            the_string = f"Individual {i}:\t Fitness= {population.individuals[i].fitness}"
            self.logger.debug(the_string)

        # Exit program if number of generation is achieved
        if self.current_generation >= self.NUMBER_OF_GENERATIONS:
            self.save_best_individual()
            pg.quit()
            sys.exit()

        # Make a new population and clear the individuals
        new_population = Population(population.population_size, self.game)
        new_population.individuals = []

        # Select the elite snakes
        elites = self.elitism(0.2, population)
        for i in range(len(elites)):
            new_population.individuals.append(Snake(self.game, elites[i].brain))
        #fitness_list = fitness_list[-len(mating_pool):]

        # Keep making new offspring till population size is achieved
        while len(new_population.individuals) < population.population_size:
            # Selection
            parents = self.roulette_wheel_selection(population.individuals, fitness_list, 2)
            # Crossover
            offspring = self.uniform_crossover(parents)
            # Mutation
            self.mutate(offspring, 0.02)
            new_population.individuals.append(offspring)

        self.current_generation += 1
        print(RED + "GENERATION " + str(self.current_generation) + "... " + RESET, end='', flush=True)
        return new_population

    def elitism(self, elite_percentage, population):
        # How many individuals are needed
        top_count = int(len(population.individuals) * elite_percentage)

        elite_individuals = population.individuals[-top_count:]
        return elite_individuals

    def calculate_population_fitness(self, population):
        print(YELLOW + "Calculating the fitness values..." + RESET)
        fitness_list = []
        for i in range(population.population_size):
            fitness = population.get_snake(i).calculate_fitness()
            fitness_list.append(fitness)
        return fitness_list
    
    def sort_population_on_fitness(self, population):
        print(YELLOW + "Sorting the population accourding to the fitness values..." + RESET)
        population.individuals.sort(key=lambda ind: ind.fitness)

    def roulette_wheel_selection(self, population, fitness_values, number_of_parents):
        # The first value will be the lowest because the list is sorted
        min_fitness = fitness_values[0]
        # Shift the values so that they are all positive
        fitness_values_shifted = np.array(fitness_values) + abs(min_fitness)
        total_fitness = np.sum(fitness_values_shifted)
        selection_probabilities = fitness_values_shifted / total_fitness

        # Chooses number_of_parents amount of individuals based on the selection_probabilities
        # replace=False makes sure it doesnt select the same individual more then once
        parents = np.random.choice(population, size=number_of_parents, p=selection_probabilities, replace=False)
        return parents.tolist()
    
    def uniform_crossover(self, parents):
        offspring = Snake(self.game, None)

        parent1_index = random.randint(0, len(parents) - 1)
        parent2_index = parent1_index
        while parent1_index == parent2_index:
            parent2_index = random.randint(0, len(parents) - 1)

        for i, layer in enumerate(offspring.brain.layers):
            for j, neuron in enumerate(layer.weights):
                for k, weight in enumerate(neuron):
                    
                    if np.random.rand() > 0.5: # tried changing this to 0.95
                        layer.weights[j, k] = parents[parent1_index].brain.layers[i].weights[j, k]
                    else:
                        layer.weights[j, k] = parents[parent2_index].brain.layers[i].weights[j, k]

            for j, bias in enumerate(layer.biases[0]):
                if np.random.rand() > 0.5:
                    layer.biases[0, j] = parents[parent1_index].brain.layers[i].biases[0, j]
                else:
                    layer.biases[0, j] = parents[parent2_index].brain.layers[i].biases[0, j]

        return offspring
    
    def mutate(self, individual, mutation_rate):
        for layer in individual.brain.layers:
            for neuron in layer.weights:
                for i, weight in enumerate(neuron):
                    if np.random.rand() <= mutation_rate:
                        neuron[i] = np.random.uniform(-1, 1) # tried changing it to lower values