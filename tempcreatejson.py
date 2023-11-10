import json, os


param_dict = {
"SIZE" : {'W':300, 'H':300},
"FPS" : 60,
"PHOTON_INTERACTIONS" : {
    "sticking_photon" :True,
    "protonaction_photon" :True,
    "neutronaction_photon" :True
},

"CONSTANTS" : {
    "particle_counting_steps": 1, # After each n steps the code gives you a statistics of number of particles

    "Probability_of_sticking": 1,

    #Defining weights for sticking; w1: the particle which incoming particle is facing, w2: two neighbors of w1, w3: two neighbors of w2s, w4: only neighbor of w3s
    "Sticking_w1_input": 1,
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
    
    "sink_size": 30
    },

'EXEC_ORDER' : [
    # 'count_particles',
    'propagation_prot_neut_step',
    'propagation_photon_step',
    'sticking_step',
    # 'scattering_step',
    #'protonaction_step',
    #'neutronaction_step',
    'absorption_step',
    # 'sink_step'
],
'PHOTON_CREATE_ORDER' : {
    'Sticking_photon': 0,
    'Protonaction_photon': 1,
    'Neutronaction_photon': 2,
}
}

with open(os.path.join(os.getcwd(), 'Config.json'), 'w') as f:
    json.dump(param_dict, f, indent=4)