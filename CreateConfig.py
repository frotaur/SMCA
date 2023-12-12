import os, json
from pathlib import Path
import random

def make_config(temporary=False):
    """
        Creates a configuration file with the defined parameters.
        Modify the parameters and run this script (python CreateConfig.py)
        to create a configuration file in the configurations/ folder.
    """
    CONFIG_NAME = 'no_photon'

    WIDTH = 300
    HEIGHT = 300
    FPS = 60

    # * If you want any of interactions create photon, set their boolean True
    photon_creations = {
    "sticking_photon" : True,
    "protonaction_photon" : False,
    "neutronaction_photon" : False,
    }

    # Specify the order in which functions should be executed. If you do not want a function to operate, you should not put its name in this list.
    #! It is sensitive to capital or small letter
    execution_order = [
        # 'count_particles',
        'sticking_step',
        'propagation_prot_neut_step',
        'propagation_photon_step',
        # 'propagation_anti_photon_step',
        #'scattering_step',
        #'protonaction_step',
        #'neutronaction_step',
        'sink_step',
        # 'photon_annihilation_step',
        'source_step',
        'absorption_step',
        # 'arbitrary_step',

    ]


    constants_dict = {
        ## REMOVE capitalization eventually

        # Set True if you want to see the photons, or False if you do not want to.
        "photon_visualization": True,

        # Set True if you want the sticking function to prefer the moving direction when there are many dominant directions, or False if you want no preference.
        "sticking_prefers_moving_direction": True,

        "particle_counting_steps": 1, # After each n steps the code gives you a statistics of number of particles

        "Probability_of_sticking": 1,

        # weight1 and weight2 are the weight of first and second neighbours respectively in scattering
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

        #Probability of absorption based on the number of neighbors moving in the same direction
        "Photon_absorption_probability_0": 1,
        "Photon_absorption_probability_1": 0,
        "Photon_absorption_probability_2": 0,
        "Photon_absorption_probability_3": 1,
        "Photon_absorption_probability_4": 1,
        "Photon_absorption_probability_5": 1,
        "Photon_absorption_probability_6": 1
        
    }

    initialization_dict = {
        # Initialization uses normal random function if 'use_random' is set True, and uses manual function if is set False. 
        'use_random': True,
        'rand_particle_threshold': 2.5,
        # Setting the initial array for protons and neutrons manually
        # first component is direction and second and third components are position of particles
        #! If you don't want to have protons or neutrons, you should define the initial numpy array and set it empty ([])
        'manual_initial_protons': [[0,2,3],[3,3,2],[3,5,2],[3,5,4],[3,3,4],[1,3,2],[2,5,2],[5,5,4],[6,3,4]],
        'manual_initial_neutrons': [[0,2,5]]
    }


    curpath = os.path.join(Path(__file__).parent.as_posix(),'configurations')

    os.makedirs(curpath,exist_ok=True)

    if(temporary):
        CONFIG_NAME = 'current'
    
    with open(os.path.join(curpath,CONFIG_NAME+'.json'),'w') as f:
        full_config = {
        'SIZE' : {'W':WIDTH,'H':HEIGHT},
        'FPS' : FPS,
        'PHOTON_CREATIONS' : photon_creations,
        'CONSTANTS' : constants_dict,
        'EXEC_ORDER' : execution_order,
        'INITIALIZATION' : initialization_dict
        }
        json.dump(full_config,f,indent=4)



if __name__ == '__main__':
    make_config()

#TODO: This function is not complete yet
def rando_config(temporary=False):
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
    photon_creations = {
    "sticking_photon" : random.choice([True,False]),
    "protonaction_photon" : random.choice([True,False]),
    "neutronaction_photon" : random.choice([True,False])
    }

    
    # NOT SURE HOW TO RANDOMIZE THIS DISCUSS WITH ALI
    execution_order = [
        'count_particles',
        'sticking_step',
        'propagation_prot_neut_step',
        'propagation_photon_step',
        #'scattering_step',
        #'protonaction_step',
        #'neutronaction_step',
        #'photon_annihilation_step',
        'absorption_step',
    ]
    random.shuffle(execution_order)

    # RANDOMIZE THIS
    constants_dict = {
        ## REMOVE capitalization eventually

        # Set True if you want to see the photons
        "photon_visualization": True,

        # Set True if you want the sticking function to prefer the moving direction when there are many dominant directions, or False if you want no preference.
        "sticking_prefers_moving_direction": True,

        "particle_counting_steps": 1, # After each n steps the code gives you a statistics of number of particles

        "Probability_of_sticking": 1,

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

        #Probability of absorption based on the number of neighbors moving in the same direction
        "Photon_absorption_probability_0": 1,
        "Photon_absorption_probability_1": 0,
        "Photon_absorption_probability_2": 0,
        "Photon_absorption_probability_3": 1,
        "Photon_absorption_probability_4": 1,
        "Photon_absorption_probability_5": 1,
        "Photon_absorption_probability_6": 1
        
    }

    initialization_dict = {
        # Initialization uses normal random function if 'use_random' is set True, and uses manual function if is set False. 
        'use_random': True,
        'rand_particle_threshold': 2.5,
        # Setting the initial array for protons and neutrons manually
        # first component is direction and second and third components are position of particles
        #! If you don't want to have protons or neutrons, you should define the initial numpy array and set it empty ([])
        'manual_initial_protons': [[0,2,3],[3,3,2],[3,5,2],[3,5,4],[3,3,4],[1,3,2],[2,5,2],[5,5,4],[6,3,4]],
        'manual_initial_neutrons': [[0,2,5]]
    }


    curpath = os.path.join(Path(__file__).parent.as_posix(),'configurations')

    os.makedirs(curpath,exist_ok=True)

    if(temporary):
        CONFIG_NAME = 'current'
    
    with open(os.path.join(curpath,CONFIG_NAME+'.json'),'w') as f:
        full_config = {
        'SIZE' : {'W':WIDTH,'H':HEIGHT},
        'FPS' : FPS,
        'PHOTON_CREATIONS' : photon_creations,
        'CONSTANTS' : constants_dict,
        'EXEC_ORDER' : execution_order,
        'INITIALIZATION' : initialization_dict
        }
        json.dump(full_config,f,indent=4)



