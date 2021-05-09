import pygame
from random import randint
from functools import reduce
import time

w, h = 800, 600
pygame.init()
screen = pygame.display.set_mode((w, h))

BLACK = pygame.Color("black")
WHITE = pygame.Color("white")
GREY = pygame.Color("grey")
PINK = pygame.Color("magenta")


def dist(a):
    return sum([((a[i][0]-a[i+1][0])**2 + (a[i][1]-a[i+1][1])**2)**0.5 for i in range(len(a)-1)])


def main():

    num_towns = 20
    towns = [(randint(w//10,9*w//10),randint(h//10,9*h//10)) for _ in range(num_towns)]
    best = [dist(towns), towns]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill(BLACK)

        for point in towns:
            pygame.draw.circle(screen, WHITE, point, 5)

        i,j = randint(0, len(towns)-1), randint(0, len(towns)-1)
        towns[i], towns[j] = towns[j], towns[i]
        d = dist(towns)
        if d < best[0]:
            best[0] = d
            best[1] = towns[:]
        pygame.draw.lines(screen, GREY, False, towns, width=1)
        pygame.draw.lines(screen, PINK, False, best[1], width=3)

        pygame.display.flip()
        time.sleep(1)


if __name__ == "__main__":
    main()