import json,os
from pathlib import Path
import numpy as np


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
    particles = np.where(particles < -threshold,-1,particles)
    particles = particles.astype(np.int16)

    return particles

def manual_particles(W,H, protons = np.ndarray, neutrons = np.ndarray):
    particles = np.zeros((7,W,H))
    for i in range(np.size(protons,axis = 0)):
        particles[protons[i][0], protons[i][1], protons[i][2]] = -1
    
    for i in range(np.size(neutrons,axis = 0)):
        particles[neutrons[i][0], neutrons[i][1], neutrons[i][2]] = 1

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
        photon_creations = config['PHOTON_CREATIONS']

        execution_order = config['EXEC_ORDER']
        constants_dict = config['CONSTANTS']
        fps = config['FPS']

        if(config['INITIALIZATION']['use_random']):
            partic_thresh = config['INITIALIZATION']['rand_particle_threshold']
            init_particles = normal_random_particles(partic_thresh,width,height)
        else:
            init_protons = np.array(config['INITIALIZATION']['manual_initial_protons'], dtype=int)
            print(f'protons= {init_protons}')
            init_neutrons = np.array(config['INITIALIZATION']['manual_initial_neutrons'], dtype=int)
            print(f'neutrons= {init_neutrons}')
            init_particles = manual_particles(width,height,init_protons,init_neutrons)
    
    return {'fixed':((width,height),fps),
            'constants' : (photon_creations,execution_order,constants_dict),
             'init' : (init_particles,)
     }

