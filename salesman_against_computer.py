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


def dist(route:np.ndarray, return_longest_segment=False):
    total = 0
    longest_segment = None
    longest_segment_index = None
    for i in range(len(route)-1):
        d = np.linalg.norm(route[i]-route[i+1])
        if not longest_segment or d > longest_segment:
            longest_segment = d
            longest_segment_index = i
        total += d
    if return_longest_segment:
        return total, longest_segment_index
    return total


def evaluate_routes(pop:list):
    best_d, prune_candidate = dist(pop[0], True)
    best_route = pop[0]
    fitnesses = []
    for i in range(1,len(pop)):
        route = pop[i]
        d, pc = dist(route, True)
        fitnesses.append(d)
        if d < best_d:
            best_d = d
            best_route = route
            prune_candidate = pc
    return {
        "fitness_list":fitnesses, 
        "best_gen_d":best_d, 
        "best_gen_route":best_route,
        "prune_candidate":prune_candidate
    }


def create_random_orders(a:list, n:int):
    random_orders = []
    for i in range(n):
        temp = a[:]
        random_orders.append([])
        while len(temp)>0:
            random_orders[i].append(temp.pop(randint(0,len(temp)-1)))
    return random_orders


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
            j = randint(0, len(genes)-1)
            genes[i], genes[j] = genes[j], genes[i]
    return genes


def prune(genes, i):
    j = randint(0,len(genes)-1)
    genes = genes[:]
    genes[i], genes[j] = genes[j], genes[i]
    return genes


def natural_selection(pop:list, _fitnesses:list, prune_candidate, do_crossover:bool=True, ):
    max_fitness = min(_fitnesses)
    fitnesses = 1 / (np.array(_fitnesses) / max_fitness)

    pool = []
    for agent, fitness in zip(pop, fitnesses):
        pool += [agent] * int(fitness*len(pop))
    pool_size = len(pool)

    children = []
    for _ in range(len(pop)-2):
        p1 = pool[randint(0,pool_size-1)]
        if do_crossover:
            p2 = pool[randint(0,pool_size-1)]
            genes = cross_over(p1,p2)
        else:
            genes = p1
        genes = mutate(genes)
        children.append(genes)
    children.append(pop[_fitnesses.index(max_fitness)])
    children.append(prune(pop[_fitnesses.index(max_fitness)], prune_candidate))

    return children


def closest_town(point, towns):
    point = np.array(point)
    closest_town = towns[0]
    for town in towns:
        if np.linalg.norm(point-town) < np.linalg.norm(point-closest_town):
            closest_town = town
    return closest_town


def main():

    num_towns = 50
    pop_size = 100
    towns = [np.array((randint(w//10,9*w//10),randint(h//10,9*h//10))) for _ in range(num_towns)]
    population = create_random_orders(towns, pop_size)

    results = evaluate_routes(population)
    best_ever_d = results["best_gen_d"]
    best_ever_route = results["best_gen_route"]

    iterations = 0
    iteration_thresh = num_towns**2
    
    time_start = time.time()
    points = []
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                closest = closest_town(pygame.mouse.get_pos(), towns)
                if not reduce(lambda a,b: a or b,[np.array_equal(closest, town) for town in points], False):
                    points.append(closest_town(pygame.mouse.get_pos(), towns))
        
        screen.fill(BLACK)
        for point in towns:
            pygame.draw.circle(screen, WHITE, point, 5, 2)
        if len(points)>1:
            pygame.draw.lines(screen, WHITE, False, points)
        if len(points)>0:
            pygame.draw.circle(screen, GREEN, points[-1], 5, 2)
        
        if len(points) == num_towns:
            running = False

        results = evaluate_routes(population)
        population = natural_selection(population, results["fitness_list"], results["prune_candidate"], True)
        if results["best_gen_d"] < best_ever_d:
            best_ever_d = results["best_gen_d"]
            best_ever_route = results["best_gen_route"]
        
        pygame.display.flip()

    time_end = time.time()
    elapsed_time = time_end-time_start
    distance = dist(points)

    population.append(points)

    start_time = time.time()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill(BLACK)

        for point in towns:
            pygame.draw.circle(screen, WHITE, point, 5, 2)

        results = evaluate_routes(population)
        pygame.draw.lines(screen, GREY, False, population[-2], width=1)

        population = natural_selection(population, results["fitness_list"], results["prune_candidate"], False)
        iterations+=1

        if results["best_gen_d"] < best_ever_d:
            best_ever_d = results["best_gen_d"]
            best_ever_route = results["best_gen_route"]
        pygame.draw.lines(screen, PINK, False, best_ever_route, width=3)

        pygame.display.flip()
        # time.sleep(1/30)

        if time.time()-start_time > elapsed_time:
            running = False

    print(distance, best_ever_d)
    if distance <= best_ever_d:
        print("YOU WIN")
    else:
        print("YOU LOSE")

if __name__ == "__main__":
    main()