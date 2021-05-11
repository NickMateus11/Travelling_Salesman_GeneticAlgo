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


def route_dist(route:np.ndarray):
    total = 0
    for i in range(len(route)-1):
        d = dist(route[i], route[i+1])
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


def main():

    num_towns = 20
    towns = [np.array((randint(w//10,9*w//10),randint(h//10,9*h//10))) for _ in range(num_towns)]
    min_d = route_dist(towns)
    best_route = towns

    iteration = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                towns = [np.array((randint(w//10,9*w//10),randint(h//10,9*h//10))) for _ in range(num_towns)]
                min_d = route_dist(towns)
                best_route = towns
                iteration = 0
        
        screen.fill(BLACK)

        for point in towns:
            pygame.draw.circle(screen, WHITE, point, 5, 2)

        towns_left = towns[:]
        route = []

        if iteration < num_towns:
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
                time.sleep(1/(2*num_towns))

            iteration += 1
            if route_d < min_d:
                min_d = route_d
                best_route = route
            pygame.draw.lines(screen, GREY, False, route, width=2)
            
        else:
            pygame.draw.lines(screen, PINK, False, best_route, width=3)

        pygame.display.flip()


if __name__ == "__main__":
    main()