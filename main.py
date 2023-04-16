import numpy as np
import matplotlib.pyplot as plt
import random


fitnesses = []

def coordinates(n_points):

    x = np.random.randint(0, 100, n_points)
    y = np.random.randint(0, 100, n_points)

    return list(zip(x, y))

def distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

#in fitness function we use the inverse of the total distance as the fitness (shorter distance = higher fitness)
def fitness(path):
    return 1/(sum([distance(path[i], path[i+1]) for i in range(len(path)-1)]) + distance(path[-1], path[0]))


# inspired by gokturk ucoluk's paper at https://user.ceng.metu.edu.tr/~ucoluk/research/publications/tspnew.pdf
def crossover(parent1, parent2):

    crossover_point = random.randint(0, len(parent1)-1)
    child = parent1[:crossover_point]

    for i in parent2:
        if i not in child:
            child.append(i)
    return child


#mutate a path by swapping two random points
def mutate(path):

    start = random.randint(0, len(path)-1)
    end = random.randint(0, len(path)-1)
    path[start], path[end] = path[end], path[start]
    return path

def selection(population, percentage):
    #select the best percentage of the population
    return sorted(population, key=fitness, reverse=True)[:int(len(population)*percentage)]

def generate_population(coordinates, population_size):

    population = []

    for i in range(population_size):
        population.append(random.sample(coordinates, len(coordinates)))

    return population

def breed(population, population_size, mutation_rate):

    next_gen = population.copy()

    for i in range(population_size-len(population)):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        next_gen.append(crossover(parent1, parent2))

        if random.random() < mutation_rate:
            next_gen[-1] = mutate(next_gen[-1])
    return next_gen

def genetic_algorithm(coordinates, population_size, generations, mutation_rate, selection_percentage):

    population = generate_population(coordinates, population_size)

    #the fitnesses list is used to plot the fitness over generations
    fitnesses.append((fitness(population[0]), 0))

    for i in range(generations):
        population = selection(population, selection_percentage)
        fitnesses.append((fitness(population[0]), i+1))
        population = breed(population, population_size, mutation_rate)

    population = selection(population, 0.5)
    return population[0]


def plot_path(path):

    x = [i[0] for i in path]
    x.append(path[0][0])

    y = [i[1] for i in path]
    y.append(path[0][1])

    plt.plot(x, y)
    plt.title("Path")
    plt.show()


#this function plots the fitness over generations
def plot_fitness():

    x = [i[1] for i in fitnesses]
    y = [i[0] for i in fitnesses]

    plt.plot(x, y)
    plt.title("Fitness over generations")
    plt.show()




path = genetic_algorithm(coordinates(int(input("choose number of cities "))),
                         int(input("choose population size ")),
                         int(input("choose number of generations ")),
                         float(input("choose mutation rate ")),
                         float(input("choose selection percentage ")))
plot_path(path)
plot_fitness()


