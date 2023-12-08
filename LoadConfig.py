import json,os
from pathlib import Path
import numpy as np
import random


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
            proton_percent = config['INITIALIZATION']['proton_percent']
            neutron_percent = config['INITIALIZATION']['neutron_percent']
            init_particles = sanitize_init_particles(percent_random_particles(proton_percent,neutron_percent,width,height).astype(np.int16))
        else:
            init_particles= sanitize_init_particles(custom_particles(width,height).astype(np.int16))
    
    return {'fixed':((width,height),fps),
            'constants' : (photon_creation_map,execution_order,constants_dict),
             'init' : (init_particles,)}

def sanitize_init_particles(init_particles):
    """
        Given an init particles tensor, removes the particles which are not on the hexagonal grid.
    """
    new_partic = init_particles.copy()
    new_partic[:,::2,::2] = 0
    new_partic[:,1::2,1::2] = 0

    return new_partic


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
