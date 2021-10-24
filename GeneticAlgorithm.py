import numpy as np
from copy import copy
import time


class GeneticAlgorithm:

    def __init__(self, matrix=[], init_population_size=100, mutation_chance=10, max_population_size=500, rate_cross = 0.2):
        self.color_names = ['red', 'blue', 'yellow', 'green']
        self.num_of_colors = len(self.color_names)

        self.init_population_size = init_population_size
        self.mutation_chance = mutation_chance
        self.max_population_size = max_population_size
        self.max_possible_score = 0
        self.adjacency_matrix = matrix
        self.num_of_regions = len(self.adjacency_matrix)
        self.rate_cross = rate_cross

    # Randomly initializing first population
    def initiate_population(self):
        population = []
        for i in range(self.init_population_size):
            population.append(list(np.random.randint(0, self.num_of_colors, self.num_of_regions)))
        return population


    # Calculating fitness score for each region set
    def calculate_fitness(self, chromosome, adjacency_matrix, maxloop):
        fitness_score = 0
        for i in range(self.num_of_regions):
            for j in range(i + 1, self.num_of_regions):
                if adjacency_matrix[i][j] == 1 and chromosome[i] == chromosome[j]:
                    fitness_score -= 10
        if fitness_score == -10  and maxloop>=0:
            self.Adjust_chromosome(chromosome, adjacency_matrix)
            return self.calculate_fitness(chromosome, adjacency_matrix, maxloop - 1)
        return fitness_score


    def Adjust_chromosome(self, chromosome, adjacency_matrix):
        print("adjust")
        x,y = -1,-1
        for i in range(self.num_of_regions):
            for j in range(i + 1, self.num_of_regions):
                if adjacency_matrix[i][j] == 1 and chromosome[i] == chromosome[j]:
                    x, y=i, j
        while True:
            chromosome[y] = np.random.randint(0, self.num_of_colors)
            if chromosome[y] != chromosome[x]:
                break




    # One point crossover
    def one_point_crossover(self, first_chromosome, second_chromosome):
        midpoint = np.random.randint(1, self.num_of_regions)
        first_set, second_set = copy(first_chromosome), copy(second_chromosome)
        first_set[midpoint:], second_set[midpoint:] = second_set[midpoint:], first_set[midpoint:]
        return first_set, second_set

    # Two point crossover
    def multi_point_crossover(self, first_chromosome, second_chromosome):
        midpoint_1 = np.random.randint(1, self.num_of_regions - 1)
        midpoint_2 = np.random.randint(midpoint_1, self.num_of_regions)
        first_set, second_set = copy(first_chromosome), copy(second_chromosome)
        first_set[midpoint_1:midpoint_2], second_set[midpoint_1:midpoint_2] \
            = second_set[midpoint_1:midpoint_2], first_set[midpoint_1:midpoint_2]
        return first_set, second_set

    # Random point crossover
    def uniform_crossover(self, first_chromosome, second_chromosome):
        first_set, second_set = copy(first_chromosome), copy(second_chromosome)
        for i in range(self.num_of_regions):
            if np.random.randint(0, 1) == 1:
                first_set[i], second_set[i] = second_set[i], first_set[i]
        return first_set, second_set

    # mutating chromosome
    def mutation(self, chromosome):
        mutated_set = copy(chromosome)
        random_index = np.random.randint(0, self.num_of_regions)
        random_color = np.random.randint(0, self.num_of_colors)
        mutated_set[random_index] = random_color
        return mutated_set

    # eliminating non-unique chromosomes
    def select_uniques(self, population):
        uniques = dict()
        for chromosome in population:
            hashable = tuple(chromosome)
            if hashable not in uniques:
                uniques[hashable] = chromosome
        return list(uniques.values())

    def genetic_algorithm(self):
        a=0
        population = self.initiate_population()
        fitness_scores = []
        #for step in range(num_of_steps):
        while True:
            a+=1
            sz = len(population)

            # Step 2
            for i in range(int(sz*self.rate_cross)):
                for j in range(i + 1, int(sz*self.rate_cross)):
                    cross_rand = np.random.randint(0, 10)
                    if cross_rand == 0:
                        child_1, child_2 = self.one_point_crossover(population[i], population[j])
                    elif cross_rand == 1:
                        child_1, child_2 = self.multi_point_crossover(population[i], population[j])
                    elif cross_rand == 2:
                        child_1, child_2 = self.uniform_crossover(population[i], population[j])
                    if cross_rand <= 2:
                        population.append(child_1)
                        population.append(child_2)

            # Step 3
            for i in range(sz):
                mutation_rand = np.random.randint(0, 100)
                if mutation_rand < self.mutation_chance:
                    mutated = self.mutation(population[i])
                    population.append(mutated)

            # Step 4
            population = self.select_uniques(population)
            population.sort(key=lambda chromo: self.calculate_fitness(chromo, self.adjacency_matrix, 3), reverse=True)
            population = population[:self.max_population_size]
            fitness_scores.append(self.calculate_fitness(population[0], self.adjacency_matrix, 3))

            # if step % print_steps == 0:
            #     print("{0}. Step | Max. Fitness: {1}".format(step, fitness_scores[-1]))
            print(fitness_scores[-1])
            if fitness_scores[-1] == self.max_possible_score:
                #print("Found optimal solution in {0} step".format(step))
                break

        best_solution = population[0]
        print("Fitness score of best solution: {0}".format(fitness_scores[-1]))
        return best_solution
        # for i in range(self.num_of_regions):
        #     print(self.color_names[best_solution[i]])
        # print(a)
def GA_coloringmap(matrix, color):
    genetic_Obj = GeneticAlgorithm(matrix)
    genetic_Obj.color_names = copy(color)
    return genetic_Obj.genetic_algorithm()


