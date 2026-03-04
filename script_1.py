import pygame
from sys import exit

pygame.init()

screen = pygame.display.set_mode((800, 400))

pygame.display.set_caption('Adaptive Agent')

icon = pygame.image.load('n_cat.jpg')
pygame.display.set_icon(icon)

# A clock object helps with time, which helps with framerate control
clock = pygame.time.Clock() 

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()               # uninitializes pygame (opposite of init)
            exit()
    # draw all elements
    # update everything
    pygame.display.update()
    clock.tick(60)              # while loop shouldn't run faster than 60 times per second
