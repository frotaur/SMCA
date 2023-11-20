import pygame
from Camera import Camera
from Automaton import *
from LoadConfig import load_config
from CreateConfig import make_config
import cv2
import os

# Select configuration name to load
# If 'None', it will use configuration currently defined in CreateConfig.py
CONFIG_NAME = None

if(CONFIG_NAME is None):
    make_config(temporary=True)
    CONFIG_NAME = 'current'



conf_fold = os.path.join(Path(__file__).parent.as_posix(),'configurations')

configo = load_config(os.path.join(conf_fold,f'{CONFIG_NAME}.json'))
((Width,Height),FPS) = configo['fixed'] # Fixed parameters
(photon_creation_map,execution_order ,constants_dict) = configo['constants'] # Automaton parameters
(init_particles,) = configo['init'] # Initialization

# Initialize the pygame screen 
pygame.init()
screen = pygame.display.set_mode((Width,Height),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
camera = Camera(Width,Height)

#Initialize the world_state array, of size (W,H,3) of RGB values at each position.
world_state = np.random.randint(0,255,(Width,Height,3),dtype=np.uint8)

# Initialize the automaton
updating = True
recording = False
launch_video = False

auto = SMCA_Triangular((Width,Height), photon_creation_map, execution_order, constants_dict, init_particles)

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
            if(event.key == pygame.K_q):
                # Here in principle we should randomize the prameters and set them
                auto.set_parameters(*(load_config(os.path.join(conf_fold,f'no_photon.json'))['constants']),init_particles=None)
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
            video_out = cv2.VideoWriter(vid_loc, fourcc, 30.0, (Width, Height))

        frame_bgr = cv2.cvtColor(auto.worldmap, cv2.COLOR_RGB2BGR)
        video_out.write(frame_bgr)
        pygame.draw.circle(surface, (255,0,0), (Width-10,Height-10),2)

    # Clear the screen
    screen.fill((0, 0, 0))
    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)
    screen.blit(zoomed_surface, (0,0))
    # Update the screen
    pygame.display.flip()
    # flip() the display to put your work on screen

    clock.tick(FPS)  # limits FPS to 60


pygame.quit()
video_out.release()