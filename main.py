import pygame, cv2
from main_utils.Camera import Camera
from Automaton import *
from LoadConfig import load_config
from CreateConfig import make_config
from main_utils.main_utils import launch_recording

# SELECT CONFIGURATION NAME TO LOAD
# If 'None', it will use configuration currently defined in CreateConfig.py
CONFIG_NAME = None

if(CONFIG_NAME is None):
    make_config(temporary=True)
    CONFIG_NAME = 'current'


# ------------------ LOAD CONFIGURATION AND INSTANCIATE AUTOMATON------------------
conf_fold = os.path.join(Path(__file__).parent.as_posix(),'configurations')
current_config = load_config(os.path.join(conf_fold,f'{CONFIG_NAME}.json')) # Load config preprocesses the configuration file
((W,H),FPS) = current_config['fixed'] # Fixed parameters
(photon_creation_map,execution_order ,constants_dict) = current_config['constants'] # Automaton parameters
(init_particles,) = current_config['init'] # Initialization

auto = SMCA_Triangular((W,H), photon_creation_map, execution_order, constants_dict, init_particles)
# ----------------------------------------------------------------------------------

# Initialize the pygame screen 
pygame.init()
screen = pygame.display.set_mode((W,H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
camera = Camera(W,H)

#Initialize the world_state array (for visualization ONLY), of size (W,H,3) of RGB values at each position.
world_state = np.zeros((W,H,3),dtype=np.uint8)

# Initialize the automaton
updating = True
recording = False
launch_video = False



while running:
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
                # The following is just a placeholder
                auto.set_parameters(*(load_config(os.path.join(conf_fold,f'random_last.json'))),init_particles=None)
        # Handle the event loop for the camera
        camera.handle_event(event)
    
    
    if(updating):
        auto.step()

    auto.draw() # Draw the automaton, updates worldmap
    #Retrieve the world_state from automaton
    world_state = auto.worldmap

    #Make the viewable surface.
    surface = pygame.surfarray.make_surface(world_state)

    #For recording
    if(recording):
        if(launch_video):
            video_out = launch_recording(W,H)
            launch_video=False
        frame_bgr = cv2.cvtColor(auto.worldmap, cv2.COLOR_RGB2BGR)
        video_out.write(frame_bgr.transpose(1,0,2))
        pygame.draw.circle(surface, (255,0,0), (W-10,H-10),2)

    # Clear the screen
    screen.fill((0, 0, 0))
    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)
    screen.blit(zoomed_surface, (0,0))
    # Update the screen
    pygame.display.flip()

    clock.tick(FPS)  # limits FPS


pygame.quit()
if(not launch_video):
    video_out.release()