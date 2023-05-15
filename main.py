import torch
import pygame
from Camera import Camera
from Automaton import *

pygame.init()
W,H = 400,400
screen = pygame.display.set_mode((W,H))
clock = pygame.time.Clock()
running = True
camera = Camera(W,H)

world_state = torch.randint(0,255,(W,H,3),dtype=torch.uint8)

auto = SMCA((W,H))

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        camera.handle_event(event)
    

    auto.step()
    world_state = auto.worldmap
    surface = pygame.surfarray.make_surface(world_state)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)

    screen.blit(zoomed_surface, (0,0))

    # Update the screen
    pygame.display.flip()
    # flip() the display to put your work on screen

    clock.tick(30)  # limits FPS to 60

pygame.quit()