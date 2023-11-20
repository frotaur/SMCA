import os, json
from pathlib import Path

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
    photon_interactions = {
    "sticking_photon" : True,
    "protonaction_photon" : False,
    "neutronaction_photon" : False,
    }

    # Order this in the order you want the functions to be executed
    photon_create_order = [
            'sticking_photon',
            'protonaction_photon',
            'neutronaction_photon'
    ]

    # Specify the order in which functions should be executed. If you do not want a function to operate, you should not put its name in this list.
    #! It is sensitive to capital or small letter
    execution_order = [
        #'count_particles',
        'propagation_prot_neut_step',
        'propagation_photon_step',
        'sticking_step',
        #'scattering_step',
        #'protonaction_step',
        #'neutronaction_step',
        'photon_annihilation_step',
        'absorption_step',
        # 'sink_step'
    ]


    constants_dict = {
        ## REMOVE capitalization eventually
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



if __name__ == '__main__':
    make_config()



















# COMMENTED THIS FOR NOW, WASN'T USED ANYWHERE SO DON'T KNOW HOW TO TRANSLATE
# #Manual array for particles
# def manual_particles(W,H, protons = np.ndarray, neutrons = np.ndarray):
#     particles = np.zeros((7,W,H))
#     for i in range(np.size(protons,axis = 0)):
#         particles[protons[i][0], protons[i][1], protons[i][2]] = -1
    
#     for i in range(np.size(neutrons,axis = 0)):
#         particles[neutrons[i][0], neutrons[i][1], neutrons[i][2]] = 1

#     return particles


# Normal_rand_particles_threshold = 2.5

# #Setting the initial array for protons and neutrons manually
# #first component is direction and second and third components are position of particles
# #! If you don't want to have protons or neutrons, you should define the initial numpy array and set it empty (np.array([]))
# Initial_protons = np.array([[0,2,3],[3,3,2],[3,5,2],[3,5,4],[3,3,4],[1,3,2],[2,5,2],[5,5,4],[6,3,4]])
# Initial_neutrons = np.array([])

# #Initial_particles = manual_particles(Width, Height, Initial_protons, Initial_neutrons)
# Initial_particles = normal_random_particles(Normal_rand_particles_threshold,Width,Height)

