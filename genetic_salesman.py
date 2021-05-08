import pygame
from random import randint
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


def dist(a):
    return sum([np.linalg.norm(a[i]-a[i+1])for i in range(len(a)-1)])


def main():

    num_towns = 5
    towns = [np.array((randint(w//10,9*w//10),randint(h//10,9*h//10))) for _ in range(num_towns)]
    best = [dist(towns), towns]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill(BLACK)

        for point in towns:
            pygame.draw.circle(screen, WHITE, point, 5)

        pygame.draw.lines(screen, GREY, False, towns, width=1)

        d = dist(towns)
        if d < best[0]:
            best[0] = d
            best[1] = towns[:]
        pygame.draw.lines(screen, PINK, False, best[1], width=3)

        pygame.display.flip()
        time.sleep(1/30)


if __name__ == "__main__":
    main()