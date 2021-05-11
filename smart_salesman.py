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


def dist(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def reverse_segment(arr, start, end):
    return arr[:start] + arr[start:end+1:][::-1] + arr[end+1:]


def route_dist(route:np.ndarray):
    total = 0
    for i in range(len(route)-1):
        d = dist(route[i],route[i+1])
        total += d
    return total


def closest_town(curr, towns):
    min_d = dist(curr, towns[0])
    index = 0
    for i,t in enumerate(towns):
        d = dist(curr, t)
        if d < min_d:
            min_d = d
            index = i
    return min_d, index


def evaluate_routes(pop:list):
    best_d = route_dist(pop[0])
    best_route = pop[0]
    fitnesses = []
    for i in range(1,len(pop)):
        route = pop[i]
        d = route_dist(route)
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


def mutate(genes):
    mutation_rate = 0.1
    for i in range(len(genes)-1):
        if random() < mutation_rate:
            genes = prune(genes, i)
    return genes


def prune(genes, i=None):
    if i is None:
        i = randint(0,len(genes)-1)
    j = randint(i+1,len(genes)-1)
    min_d = route_dist(genes)
    min_g = genes

    while j < len(genes):
        g = reverse_segment(genes,i,j)
        d = route_dist(g)
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
        children.append(mutate(pool[randint(0,pool_size-1)]))
    children.append(pop[_fitnesses.index(max_fitness)])

    return children


def main():

    num_towns = 50
    pop_size = 10
    towns = [np.array((randint(w//10,9*w//10),randint(h//10,9*h//10))) for _ in range(num_towns)]
    population = create_random_orders(towns, pop_size)

    results = evaluate_routes(population)
    best_ever_d = results["best_gen_d"]
    best_ever_route = results["best_gen_route"]

    for _ in range(max(num_towns//2,10)):
        iteration = randint(0,num_towns-1)
        screen.fill(BLACK)

        for point in towns:
            pygame.draw.circle(screen, WHITE, point, 5, 2)
        
        towns_left = towns[:]
        route = []

        route.append(towns[iteration])
        towns_left.pop(iteration)
        route_d = 0
        curr_town = towns[iteration]
        while len(towns_left) > 0:
            d, i = closest_town(curr_town, towns_left)
            route_d += d
            curr_town = towns_left[i]
            route.append(towns_left[i])
            towns_left.pop(i)

            screen.fill(BLACK)
            pygame.draw.lines(screen, GREY, False, route, width=2)
            for point in towns:
                pygame.draw.circle(screen, WHITE, point, 5, 2)
            pygame.draw.circle(screen, GREEN, towns[iteration], 5, 2)
            pygame.display.flip()
            time.sleep(1/(num_towns/2)**2)

        if route_d < best_ever_d:
            best_ever_d = route_d
            best_ever_route = route
    
        pygame.display.flip()

    population[-1] = best_ever_route


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

        if results["best_gen_d"] < best_ever_d:
            best_ever_d = results["best_gen_d"]
            best_ever_route = results["best_gen_route"]
        pygame.draw.lines(screen, PINK, False, best_ever_route, width=3)

        pygame.display.flip()
        # time.sleep(1/30)


if __name__ == "__main__":
    main()