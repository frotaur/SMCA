import pygame
from Camera import Camera
from Automaton import *

# Initialize the pygame screen 
pygame.init()
W,H =300,300
screen = pygame.display.set_mode((W,H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
camera = Camera(W,H)

#Initialize the world_state array, of size (W,H,3) of RGB values at each position.
world_state = np.random.randint(0,255,(W,H,3),dtype=np.uint8)

# Initialize the automaton
auto = SMCA((W,H))

updating = True
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        # Event loop. Here we deal with all the interactivity
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if(event.key == pygame.K_p):
                updating=not updating
        # Handle the event loop for the camera
        camera.handle_event(event)
    
    if(updating):
        # Step the automaton if we are updating
        auto.step()
    #Retrieve the world_state from automaton
    world_state = auto.worldmap

    #Make the viewable surface.
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