import pygame
from Camera import Camera
from Automaton import *
import cv2
import os

# Clear terminal
os.system('cls')
# Initialize the pygame screen 
pygame.init()
W, H = 1800, 900
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

cnt = -1
t_event = t_step = t_worldmap = t_surfacegen = t_video = t_render = t_total = 0

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    t0 = t_total_0 = time.time()
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
    t_event += time.time() - t0

    t0 = time.time()
    if(updating):
        # Step the automaton if we are updating
        auto.step()    
    t_step += time.time() - t0

    #Retrieve the world_state from automaton
    t0 = time.time()
    world_state = auto.worldmap
    t_worldmap += time.time() - t0

    #Make the viewable surface.
    t0 = time.time()
    surface = pygame.surfarray.make_surface(world_state)
    t_surfacegen += time.time() - t0

    #For recording
    t0 = time.time()
    if(recording):
        if(launch_video):
            launch_video = False
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            vid_loc = 'Videos/lgca1.mkv'
            video_out = cv2.VideoWriter(vid_loc, fourcc, 30.0, (W, H))

        frame_bgr = cv2.cvtColor(auto.worldmap, cv2.COLOR_RGB2BGR)
        video_out.write(frame_bgr)
        pygame.draw.circle(surface, (255,0,0), (W-10,H-10),2)
    t_video += time.time() - t0

    t0 = time.time()
    # Clear the screen
    screen.fill((0, 0, 0))
    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)
    screen.blit(zoomed_surface, (0,0))
    # Update the screen
    pygame.display.flip()
    # flip() the display to put your work on screen
    clock.tick(30)  # limits FPS to 60
    # clock.tick(10)  # limits FPS to 60
    t_render += time.time() - t0

    cnt += 1

    t_total += time.time() - t_total_0

    if cnt % SMCA.nsteps == 0:
        print("="*80)
        print("Step # = " + str(cnt))
        print("t_event = " + str(t_event))
        print("t_step = " + str(t_step))
        print("t_worldmap = " + str(t_worldmap))
        print("t_surfacegen = " + str(t_surfacegen))
        print("t_video = " + str(t_video))
        print("t_render = " + str(t_render))
        print("t_total = " + str(t_total))
        t_event = t_step = t_worldmap = t_surfacegen = t_video = t_render = t_total = 0

pygame.quit()
video_out.release()