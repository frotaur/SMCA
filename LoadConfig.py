import json,os
from pathlib import Path
import numpy as np
import random


def normal_random_particles(threshold,W,H):
    """
        Generates evenly distributed random particles with a threshold
        << UP NEXT, ADD MORE PARAMETERS, like percent of particles of each type >>
    """
    rand_array = np.random.randn(7,W,H)
    particles= np.zeros((7,W,H))
    particles[:,1::2, ::2] = rand_array[:,1::2, ::2]
    particles[:,::2, 1::2] = rand_array[:,::2, 1::2]

    tmp1 = particles <= threshold
    tmp2 = particles >= -threshold
    tmp3 = tmp1 * tmp2
    particles = np.where(tmp3,0,particles)
    particles = np.where(particles > threshold,-1,particles)
    particles = np.where(particles < -threshold,1,particles)
    particles = particles.astype(np.int16)

    return particles

def create_center_square(W, H, block_size):
    """
    Creates a grid with a square block of '-1' particles in the center on a hexagonal grid.
    """
    particles = np.zeros((7, W, H), dtype=np.float)

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
        photon_interactions = config['PHOTON_INTERACTIONS']
        photon_create_order = {config['PHOTON_CREATE_ORDER'][k]:k for k in range(len(config['PHOTON_CREATE_ORDER']))}
        photon_creation_bools = [photon_interactions[photon] for photon in photon_create_order]
        photon_creation_map = {'order':photon_create_order,'bools':photon_creation_bools} # This is fed into automaton

        execution_order = config['EXEC_ORDER']
        constants_dict = config['CONSTANTS']
        fps = config['FPS']

        if(config['INITIALIZATION']['use']):
            partic_thresh = config['INITIALIZATION']['rand_particle_threshold']
            init_particles = create_center_square(partic_thresh,width,height)
        else:
            init_particles=None
    
    return {'fixed':((width,height),fps),
            'constants' : (photon_creation_map,execution_order,constants_dict),
             'init' : (init_particles,)
     }

def rando_config(width,height,fps,temporary=False):
    """
        Creates a configuration file with the random parameters.
        STILL NEEDS WORK !

        TO BE IMPLEMENTED STILL !!!!!!
    """
    CONFIG_NAME = 'random_last'

    WIDTH = 300
    HEIGHT = 300
    FPS = 60

    # * If you want any of interactions create photon, set their boolean True
    photon_interactions = {
    "sticking_photon" : random.choice([True,False]),
    "protonaction_photon" : random.choice([True,False]),
    "neutronaction_photon" : random.choice([True,False])
    }

    # Order this in the order you want the functions to be executed
    photon_create_order = [
            'sticking_photon',
            'protonaction_photon',
            'neutronaction_photon'
    ]
    random.shuffle(photon_create_order)
    
    # NOT SURE HOW TO RANDOMIZE THIS DISCUSS WITH ALI
    execution_order = [
        #'count_particles',
        'propagation_prot_neut_step',
        'propagation_photon_step',
        'sticking_step',
        #'scattering_step',
        #'protonaction_step',
        #'neutronaction_step',
        'absorption_step',
        # 'sink_step'
    ]
    random.shuffle(execution_order)

    # RANDOMIZE THIS
    constants_dict = {
        ## REMOVE capitalization eventually
        "particle_counting_steps": 1, # After each n steps the code gives you a statistics of number of particles

        "Probability_of_sticking": 1,

        #Defining weights for sticking; w1: the particle which incoming particle is facing, w2: two neighbors of w1, w3: two neighbors of w2s, w4: only neighbor of w3s
        "Sticking_w1_input": 2,
        "Sticking_w2_input": 1,
        "Sticking_w3_input": 0,
        "Sticking_w4_input": 0,

        #Thresholds for deciding whether sticking happens or not; if the frequency of only one direction is more than threshold, then sticking happens.
        #These numbers should always be considered as a number out of 6 (since we normalized our weights)
        "Sticking_move_to_move_threshold": 4,
        "Sticking_move_to_rest_threshold": 4,
        "Sticking_rest_to_move_threshold": 4,

        #weight1 and weight2 are the weight of first and second neighbours respectively in scattering
        "Scattering_weight1": 1,
        "Scattering_weight2": 1,
        # The probability when there is no neighbor moving in the same direction of the scattered particle
        "Probability of scattering": 0.5,
        # The threshold after which the probability of scattering decreases
        "Scattering_threshold_one": 3,
        # The threshold after which the probability of scattering is zero
        "Scattering_threshold_two": 7,

        #weight1 and weight2 are the weight of first and second proton/neurton neighbours respectively
        "Prot_Neut_weight1": 1,
        "Prot_Neut_weight2": 0.5,

        #Threshold after which there is a probability that proton/neutron suddenly changes its direction
        "Prot_Neut_threshold": 6,
        #slope of the line of the linear increase of probability after threshold
        "Prot_Neut_slope": 0.3,

        "Photon_absorption_probability": 0.5,
        
        "sink_size": 30,
    }

    initialization_dict = {
        'use': True,
        'rand_particle_threshold': 2.5
    }


    curpath = os.path.join(Path(__file__).parent.as_posix(),'configurations')

    os.makedirs(curpath,exist_ok=True)

    if(temporary):
        CONFIG_NAME = 'current'
    
    with open(os.path.join(curpath,CONFIG_NAME+'.json'),'w') as f:
        full_config = {
        'SIZE' : {'W':WIDTH,'H':HEIGHT},
        'FPS' : FPS,
        'PHOTON_INTERACTIONS' : photon_interactions,
        'PHOTON_CREATE_ORDER' : photon_create_order,
        'CONSTANTS' : constants_dict,
        'EXEC_ORDER' : execution_order,
        'INITIALIZATION' : initialization_dict
        }
        json.dump(full_config,f,indent=4)

