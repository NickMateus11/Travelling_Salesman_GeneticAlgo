import pygame
from random import randint, random
from functools import reduce
from itertools import permutations
import time
import numpy as np

w, h = 800, 600
pygame.init()
screen = pygame.display.set_mode((w, h))

BLACK = pygame.Color("black")
WHITE = pygame.Color("white")
GREY = pygame.Color("grey")
PINK = pygame.Color("magenta")
GREEN = pygame.Color("green")


def reverse_segment(arr, start, end):
    return arr[:start] + arr[start:end+1:][::-1] + arr[end+1:]


def dist(route:np.ndarray):
    total = 0
    for i in range(len(route)-1):
        d = np.linalg.norm(route[i]-route[i+1])
        total += d
    return total


def evaluate_routes(pop:list):
    best_d = dist(pop[0])
    best_route = pop[0]
    fitnesses = []
    for i in range(1,len(pop)):
        route = pop[i]
        d = dist(route)
        fitnesses.append(d)
        if d < best_d:
            best_d = d
            best_route = route
    return {
        "fitness_list":fitnesses, 
        "best_gen_d":best_d, 
        "best_gen_route":best_route,
    }


def create_random_orders(a:list, n:int):
    random_routes = []
    for i in range(n):
        temp = a[:]
        random_route = []
        while len(temp)>0:
            random_route.append(temp.pop(randint(0,len(temp)-1)))
        random_routes.append(random_route)
    return random_routes


def cross_over(parent1:list, parent2:list):
    first_half = randint(0,1)
    chop = randint(0,len(parent1))
    if first_half:
        child = parent1[:chop]
    else:
        child = parent1[-chop:]

    for gene in parent2:
        insert=True
        for town in child:
            if np.array_equal(gene, town):
                insert = False
                break
        if insert:
            child.append(gene)

    return child


def mutate(genes):
    mutation_rate = 3/len(genes)
    genes = genes[:]
    for i in range(len(genes)-1):
        if random() < mutation_rate:
            # genes[i], genes[i+1] = genes[i+1], genes[i]
            genes = prune(genes, i)
    return genes


def prune(genes, i=None):
    if i is None:
        i = randint(0,len(genes)-1)
    j = randint(0,len(genes)-1)
    min_d = dist(genes)
    min_g = genes

    while j < len(genes):
        g = reverse_segment(genes,i,j)
        d = dist(g)
        if d < min_d:
            min_d = d
            min_g = g 
        j+=1
    return min_g


def natural_selection(pop:list, _fitnesses:list, do_crossover:bool=True, ):
    max_fitness = min(_fitnesses)
    fitnesses = 1 / (np.array(_fitnesses) / max_fitness)

    pool = []
    for agent, fitness in zip(pop, fitnesses):
        pool += [agent] * int(fitness*len(pop))
    pool_size = len(pool)

    children = []
    for _ in range(len(pop)-1):
        p1 = pool[randint(0,pool_size-1)]
        if do_crossover:
            p2 = pool[randint(0,pool_size-1)]
            genes = cross_over(p1,p2)
        else:
            genes = p1
        genes = mutate(genes)
        children.append(genes)
    children.append(pop[_fitnesses.index(max_fitness)])
    # children.append(prune(pop[_fitnesses.index(max_fitness)]))

    return children


def main():

    num_towns = 20
    pop_size = 20
    towns = [np.array((randint(w//10,9*w//10),randint(h//10,9*h//10))) for _ in range(num_towns)]
    population = create_random_orders(towns, pop_size)

    results = evaluate_routes(population)
    best_ever_d = results["best_gen_d"]
    best_ever_route = results["best_gen_route"]

    iterations = 0
    iteration_thresh = num_towns**2

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                towns = [np.array((randint(w//10,9*w//10),randint(h//10,9*h//10))) for _ in range(num_towns)]
                population = create_random_orders(towns, pop_size)

                results = evaluate_routes(population)
                best_ever_d = results["best_gen_d"]
                best_ever_route = results["best_gen_route"]
        
        screen.fill(BLACK)

        for point in towns:
            pygame.draw.circle(screen, WHITE, point, 5, 2)

        results = evaluate_routes(population)
        pygame.draw.lines(screen, GREY, False, population[-2], width=1)

        population = natural_selection(population, results["fitness_list"], False)
        iterations+=1

        if results["best_gen_d"] < best_ever_d:
            best_ever_d = results["best_gen_d"]
            best_ever_route = results["best_gen_route"]
        pygame.draw.lines(screen, PINK, False, best_ever_route, width=3)

        pygame.display.flip()
        # time.sleep(1/30)


if __name__ == "__main__":
    main()