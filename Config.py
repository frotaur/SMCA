import numpy as np

# * If you want any of interactions create photon, set their boolean True
Sticking_photon = True
Protonaction_photon = True
Neutronaction_photon = True

#This dictionary is created to map photon creation booleans order to their name
photon_creation_map = {
    'Sticking_photon': 0,
    'Protonaction_photon': 1,
    'Neutronaction_photon': 2,
    #! The order of booleans related to functions has to be in correspondence with above numbers
    'config_list' : [Sticking_photon,Protonaction_photon,Neutronaction_photon]
}

# Specify the order in which functions should be executed. If you do not want a function to operate, you should not put its name in this list.
#! It is sensitive to capital or small letter
execution_order = [
    # 'count_particles',
    'propagation_prot_neut_step',
    'propagation_photon_step',
    'sticking_step',
    # 'scattering_step',
    #'protonaction_step',
    #'neutronaction_step',
    'absorption_step',
    # 'sink_step'
]

Width = 300
Height = 300

FPS = 60

constants_dict = {
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
    
    "sink_size": 30,
}

#Random array for particles
def normal_random_particles(threshold,W,H):
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

#Manual array for particles
def manual_particles(W,H, protons = np.ndarray, neutrons = np.ndarray):
    particles = np.zeros((7,W,H))
    for i in range(np.size(protons,axis = 0)):
        particles[protons[i][0], protons[i][1], protons[i][2]] = -1
    
    for i in range(np.size(neutrons,axis = 0)):
        particles[neutrons[i][0], neutrons[i][1], neutrons[i][2]] = 1

    return particles


Normal_rand_particles_threshold = 2.5

#Setting the initial array for protons and neutrons manually
#first component is direction and second and third components are position of particles
#! If you don't want to have protons or neutrons, you should define the initial numpy array and set it empty (np.array([]))
Initial_protons = np.array([[0,2,3],[3,3,2],[3,5,2],[3,5,4],[3,3,4],[1,3,2],[2,5,2],[5,5,4],[6,3,4]])
Initial_neutrons = np.array([])

#Initial_particles = manual_particles(Width, Height, Initial_protons, Initial_neutrons)
Initial_particles = normal_random_particles(Normal_rand_particles_threshold,Width,Height)

