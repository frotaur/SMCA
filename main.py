import pygame
from Camera import Camera
from Automaton import *
import cv2
import os

os.system('clear')

# Initialize the pygame screen 
pygame.init()
W, H = 300, 300
screen = pygame.display.set_mode((W,H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
camera = Camera(W,H)

#Initialize the world_state array, of size (W,H,3) of RGB values at each position.
world_state = np.random.randint(0,255,(W,H,3),dtype=np.uint8)

# Initialize the automaton
auto = SMCA((W,H))
# auto = SMCA((W,H), False) # Disables clump counting

updating = True
recording = False
launch_video = False

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
            if(event.key == pygame.K_r):
                recording= not recording
                if(not launch_video):
                    launch_video=True
        # Handle the event loop for the camera
        camera.handle_event(event)

    if(updating):
        # Step the automaton if we are updating
        auto.step()    

    #Retrieve the world_state from automaton
    world_state = auto.worldmap

    #Make the viewable surface.
    surface = pygame.surfarray.make_surface(world_state)

    #For recording
    if(recording):
        if(launch_video):
            launch_video = False
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            vid_loc = 'Videos/lgca1.mkv'
            video_out = cv2.VideoWriter(vid_loc, fourcc, 30.0, (W, H))

        frame_bgr = cv2.cvtColor(auto.worldmap, cv2.COLOR_RGB2BGR)
        video_out.write(frame_bgr)
        pygame.draw.circle(surface, (255,0,0), (W-10,H-10),2)

    # Clear the screen
    screen.fill((0, 0, 0))
    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)
    screen.blit(zoomed_surface, (0,0))
    # Update the screen
    pygame.display.flip()
    # flip() the display to put your work on screen

    clock.tick(60)  # limits FPS to 60


pygame.quit()
video_out.release()