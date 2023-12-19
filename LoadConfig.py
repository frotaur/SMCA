import json,os
from pathlib import Path
import numpy as np
import random

def load_config(path):
    """
        Loading config by path. If path is relative, assumes it is
        relative w.r.t. to launched script.
    """
    curpath = Path(__file__).parent.as_posix()
    if not os.path.isabs(path):
        path = os.path.join(curpath, path)
    
    if(not os.path.exists(path)):
        raise FileNotFoundError(f"Config file not found at {path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
        width,height = config['SIZE']['W'],config['SIZE']['H']
        photon_creations = config['PHOTON_CREATIONS']

        execution_order = config['EXEC_ORDER']
        constants_dict = config['CONSTANTS']
        fps = config['FPS']

        if(config['INITIALIZATION']['use_random']):
            proton_percent = config['INITIALIZATION']['proton_percent']
            neutron_percent = config['INITIALIZATION']['neutron_percent']
            init_particles = sanitize_init_particles(percent_random_particles(proton_percent,neutron_percent,width,height).astype(np.int16))
        else:
            #Option 1
            init_protons = np.array(config['INITIALIZATION']['manual_initial_protons'], dtype=int)
            print(f'protons= {init_protons}')
            init_neutrons = np.array(config['INITIALIZATION']['manual_initial_neutrons'], dtype=int)
            print(f'neutrons= {init_neutrons}')
            init_particles = manual_particles(width,height,init_protons,init_neutrons)
            
            # TODO: We should choose between option 1 and 2
            # Option 2
            # init_particles= sanitize_init_particles(custom_particles(width,height).astype(np.int16))

    
    return {'fixed':((width,height),fps),
            'constants' : (photon_creations,execution_order,constants_dict),
             'init' : (init_particles,)
     }




def random_config(width,height,fps):
    """
        Creates a configurartion dictionary with random parameters.
        ! Not sure, but I think it is better to not craete .json file here, we can create another function for that. 
    """
    CONFIG_NAME = 'random_last'

    WIDTH = width
    HEIGHT = height
    FPS = fps

    
    
    # * If you want any of interactions create photon, set their boolean True
    photon_creations = {
    "sticking_photon" : random.choice([True,False]),
    "protonaction_photon" : random.choice([True,False]),
    "neutronaction_photon" : random.choice([True,False])
    }
    
    # Here we randomize all of our steps in one iteration(execuation order).
    # execution_order includes 3 blocks: constraints_block, interaction_block, propag_and_photon_jobs_block. 
        #constraints_block: contains the conditions imposed on the lattice, like boundary conditions, sink,source, etc. Their order is shuffled.
        #interaction_block: contains sticking, scattering, protonaction, neutronaction, absorption. each of them can be whether exists or not and their order is shuffled.
        #propag_and_photon_jobs_block: contains proton and neuron propagation, anitphoton/photon propagation, phton annihilation, and photon absorption. I recommend to not shuffle this block.
            #proton and neutron propagation should always exists, and it is done only once in each iteration.
            #antiphoton/photon propagation should always exists, and it can be done multiple times in each iteration.(the number of times is random)
            #photon annihilation can exist or not, and it is done only once in each iteration if it exists.(its eexistence is random)
            #photon absorption must always exist, since without it the existence of photon is meaningless. It is done only once in each iteration.
        # ! The order of the blocks should not be changed, except you know what you are doing.
    
    # creating constraints_block
    constraints_block = ['sink_step','source_step','arbitrary_constraints_step']
    random.shuffle(constraints_block)
    constraints_block = constraints_block[0:np.random.randint(0,len(constraints_block)+1)]
    # TODO: All of our sink, source and arbitrary constraints are place holders. We should discuss about how to implement them.
    # creating interaction_block
    interaction_block = ['sticking_step','scattering_step','protonaction_step','neutronaction_step']
    random.shuffle(interaction_block)
    interaction_block = interaction_block[0:np.random.randint(1,len(interaction_block)+1)]
    
    # creating propag_and_photon_jobs_block
    is_anitphoton_photon_propagation = random.choice([True,False])
    is_photon_annihilation = random.choice([True])
    maximum_number_of_photon_propagation = 3
    propag_and_photon_jobs_block = ['propagation_prot_neut_step']
    if is_anitphoton_photon_propagation == False: 
        for _ in range(np.random.randint(1,maximum_number_of_photon_propagation+1)):
            propag_and_photon_jobs_block.append('propagation_photon_step')
    else:
        for _ in range(np.random.randint(1,maximum_number_of_photon_propagation+1)):
            propag_and_photon_jobs_block.append('propagation_anti_photon_step')
    if is_photon_annihilation == True:
        propag_and_photon_jobs_block.append('photon_annihilation_step')
    propag_and_photon_jobs_block.append('absorption_step')

    #combining all blocks
    execution_order = constraints_block + interaction_block + propag_and_photon_jobs_block
    
       

    # Recepie for randomizing constants_dict:
        # 1. Separate constants into 2 gropus: randomizable and non-randomizable.
        # 2. Randomize the randomizable constants.(to do so, you can randomize them outside of the dictionary and then add them to the dictionary)
    constants_dict = {
        # *non-randomizable constants:
       
        "photon_visualization": False,   # Set True if you want to see the photons, or False if you do not want to.
        "particle_counting_steps": 50,  # After each n steps the code gives you a statistics of number of particles
        "Scattering_threshold_one": 3,  # The threshold after which the probability of scattering decreases linearly
        "Scattering_threshold_two": 8,  # The threshold after which the probability of scattering is zero

        # *randomizable constants:
        "sticking_prefers_moving_direction": random.choice([True,False]),   # Set True if you want the sticking function to prefer the moving direction when there are many dominant directions.
        "Probability_of_sticking": np.random.uniform(0,1),  # The probability of sticking when a direction is dominant among the neighbours
        "Scattering_weight1": np.random.uniform(0,1),   # weight1 and weight2 are the weight of first and second neighbours respectively in scattering
        "Scattering_weight2": np.random.uniform(0,1),
        "Probability of scattering": np.random.uniform(0,1),    # The probability of scattering when there is no neighbor moving in the same direction of the scattered particle
        "Prot_Neut_weight1": np.random.uniform(0,1),  #weight1 and weight2 are the weight of first and second proton/neurton neighbours respectively
        "Prot_Neut_weight2": np.random.uniform(0,1),
        "Prot_Neut_threshold": np.random.randint(0,10),   #Threshold after which there is a probability that proton/neutron suddenly changes its direction
        "Prot_Neut_slope": np.random.uniform(0,2),   #slope of the line of the linear increase of probability after threshold

        #Probability of photon absorption based on the number of neighbors moving in the same direction
        "Photon_absorption_probability_0": np.random.uniform(0,1),
        "Photon_absorption_probability_1": np.random.uniform(0,1),
        "Photon_absorption_probability_2": np.random.uniform(0,1),
        "Photon_absorption_probability_3": np.random.uniform(0,1),
        "Photon_absorption_probability_4": np.random.uniform(0,1),
        "Photon_absorption_probability_5": np.random.uniform(0,1),
        "Photon_absorption_probability_6": np.random.uniform(0,1),
        
    }
    
    #randomizing initial particles
    proton_percent = np.random.uniform(0,0.4)
    neutron_percent = np.random.uniform(0,0.4)
    init_particles = sanitize_init_particles(percent_random_particles(proton_percent,neutron_percent,WIDTH,HEIGHT).astype(np.int16))

    
    
    
    
    curpath = os.path.join(Path(__file__).parent.as_posix(),'configurations')

    os.makedirs(curpath,exist_ok=True)

    with open(os.path.join(curpath,CONFIG_NAME+'.json'),'w') as f:
        full_config = {
        'SIZE' : {'W':WIDTH,'H':HEIGHT},
        'FPS' : FPS,
        'PHOTON_CREATIONS' : photon_creations,
        'EXEC_ORDER' : execution_order,
        'CONSTANTS' : constants_dict,
        'INITIALIZATION' : {'PROTON_PERCENT': proton_percent, 'NEUTRON_PERCENT': neutron_percent},
        }
        json.dump(full_config,f,indent=4)
    
    
    
    
    return {'fixed':((WIDTH,HEIGHT),FPS),
            'constants' : (photon_creations,execution_order,constants_dict),
             'init' : (init_particles,)
     }
    

        



def percent_random_particles(proton_percent,neutron_percent,W,H):
    """
        Generates evenly distributed random particles, with a given percentage of protons and neutrons.
        Percentages are relative to the total number of sites. So there can be up to 700% of particles,
        since there are 7 directions (7 particles per site). Directions are distributed randomly.
    """
    assert proton_percent/7. + neutron_percent/7. <= 1, "Total percentage of proton and neutron times 7 must be less than 1!"
    print(f'try with {proton_percent,neutron_percent,W,H}')
    rand_array = np.random.rand(7,W,H)

    protons = rand_array <= proton_percent/7.
    neutrons = (proton_percent/7. <= rand_array) & (rand_array <= proton_percent/7. + neutron_percent/7.)

    particles = np.zeros((7,W,H),dtype=np.int16)
    particles = np.where(protons,-1,particles)
    particles = np.where(neutrons,1,particles)

    particles = particles.astype(np.int16)

    return particles


def sanitize_init_particles(init_particles):
    """
        Given an init particles tensor, removes the particles which are not on the hexagonal grid.
    """
    new_partic = init_particles.copy()
    new_partic[:,::2,::2] = 0
    new_partic[:,1::2,1::2] = 0

    return new_partic


def create_center_square(W, H, block_size):
    """
    Creates a grid with a square block of '-1' particles in the center on a hexagonal grid.
    """
    particles = np.zeros((7, W, H), dtype=np.int)

    # Calculate the starting and ending indices for the square block
    start_x = W // 2 - block_size // 2
    end_x = start_x + block_size
    start_y = H // 2 - block_size // 2
    end_y = start_y + block_size

    # Set the values in the block to -1, accounting for hexagonal grid offset
    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            if (x + y) % 2 == 0:  # Adjust for hexagonal grid offset
                particles[:, x, y] = -1

    return particles


        
        
# TODO: custom particles and manual particles should be merged into one function
def custom_particles(W,H):
    """
        Custom function for initialization state. It should return a particles tensor of integers,
        of size (7,W,H) where 7 is the number of directions. -1 for proton, 1 for neutron, 0 for nothing. TODO : CHECK THIS.`
        There are additional restriction, due to the data representation. Only (x,y) locations which will be considered will be
        x odd, y even, and x even, y odd. This is due to the hexagonal grid representation. You can just fill it liberally, but 
        these locations will be set to zero anyway.

        Returns : (7,W,H) initialization of protons and neutrons.
    """
    # PLACEHOLDER IMPLEMENTATION :
    particles = np.randint(-1,2,(7,W,H))

    return particles

def manual_particles(W,H, protons = np.ndarray, neutrons = np.ndarray):
    particles = np.zeros((7,W,H))
    for i in range(np.size(protons,axis = 0)):
        particles[protons[i][0], protons[i][1], protons[i][2]] = -1

    for i in range(np.size(neutrons,axis = 0)):
        particles[neutrons[i][0], neutrons[i][1], neutrons[i][2]] = 1

    return particles