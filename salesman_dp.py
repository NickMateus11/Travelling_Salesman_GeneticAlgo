import pygame
from random import randint, random
from functools import reduce
from itertools import permutations
import time
import numpy as np
from Held_Karp import held_karp as tsp

w, h = 800, 600
pygame.init()
screen = pygame.display.set_mode((w, h))

BLACK = pygame.Color("black")
WHITE = pygame.Color("white")
GREY = pygame.Color("grey")
PINK = pygame.Color("magenta")
GREEN = pygame.Color("green")


def tour_dist(a:np.ndarray):
    return sum([np.linalg.norm(a[i]-a[i+1])for i in range(len(a)-1)])


def create_dist_matrix(points):
    mat = np.zeros(shape=(len(points),len(points)))
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            mat[i][j] = mat[j][i] = np.linalg.norm(points[i]-points[j])
    return mat


def closest_town(point, towns):
    point = np.array(point)
    closest_town = towns[0]
    for town in towns:
        if np.linalg.norm(point-town) < np.linalg.norm(point-closest_town):
            closest_town = town
    return closest_town


def main():

    num_towns = 15
    towns = [np.array((randint(w//10,9*w//10),randint(h//10,9*h//10))) for _ in range(num_towns)]

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

        pygame.display.flip()

    distance = tour_dist(points)

    dist = create_dist_matrix(towns)
    d, town_i = tsp(dist)
    temp = town_i[:]
    for i in range(len(town_i)):
        curr_d = tour_dist([towns[j] for j in (town_i[i:] + town_i[:i])])
        if curr_d < d:
            d = curr_d
            temp = town_i[i:] + town_i[:i]
    town_i = temp

    tour = []
    for i in town_i:
        tour.append(towns[i])

    print(distance, d)
    if distance <= d:
        print("YOU WIN")
    else:
        print("YOU LOSE")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                pass
        
        screen.fill(BLACK)

        for t in towns:
            pygame.draw.circle(screen, WHITE, t, 5, 1)
        pygame.draw.lines(screen, PINK, False, tour, width=3)

        pygame.display.flip()
        # time.sleep(1/30)

if __name__ == "__main__":
    main()