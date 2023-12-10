import numpy as np
from numba import njit, prange,cuda 
from CreateConfig import *
import numba.typed as numt
import csv
import datetime


class Automaton :
    """
        Class that internalizes the rules and evolution of 
        the cellular automaton at hand. It has a step function
        that makes one timestep of the evolution. By convention,
        and to keep in sync with pygame, the world tensor has shape
        (W,H,3). It contains float values between 0 and 1, which
        are (automatically) mapped to 0 255 when returning output, 
        and describes how the world is 'seen' by an observer.

        Parameters :
        size : 2-Tuple (W,H)
            Shape of the CA world
        
    """

    def __init__(self,size):
        self.w, self.h  = size
        self.size = size
        # ! This self._worldmap should be changed in the step function.
        # It should contains floats from 0 to 1 of RGB values.
        self._worldmap = np.random.uniform(size=(self.w,self.h,3)) # W, H, 3
    

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')
    
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)
    

class SMCA_Triangular(Automaton):
    """
        Standard Model Cellular Automaton for the triangular lattice. Inspired by LGCA.

        Parameters :
                    First argument: (W,H) tuple for the size of cellur automaton ""Note: W,H must be even numbers.""
                    Second argument: Boolean. It is by default True and If you put False it does not give you the statistics. 
    """

    def __init__(self, size, photon_creations, execution_order, constants, Init_particles): # size = (W,H) "Note: W,H must be even numbers." Configuration is a list of booleans. Constants is a dictionary.
        super().__init__(size)
        self.steps_cnt = 0
        # 0,1,2,3,4,5 of the first dimension are the six directions, starting with East, and continuing clockwise
        
        self.set_parameters(photon_creations, execution_order, constants, Init_particles)
        #creating an array for photons
        self.photons = np.zeros((6,self.w,self.h),dtype=int) #dimensions of this array are the same with particles and the values of it are the number of photons in that location and direction
        
        self.dir = np.array([[2,0],[1,1],[-1,1],[-2,0],[-1,-1],[1,-1]])  # Contains arrays of the six directions, starting with East, and continuing clockwise
    
    
    def set_parameters(self, photon_creations, execution_order, constants, init_particles):
        #create a lattice in which there are some neutrons and protons. 0: nothing, -1: proton, 1: neutron
        if(init_particles is not None):
            self.particles=init_particles

        self.photon_creations = photon_creations
        self.execution_order = execution_order
        self.constants = constants
        
        # This part creates a csv file for the number of particles:
        if 'count_particles' in self.execution_order:
            self.particle_counting_nsteps = self.constants["particle_counting_steps"]   # After each n steps the code gives you a statistics of number of particles
            self.relative_path = "./CSV/"   #The name of folder in which csv files willl be saved  #! You must have a folder with the same name in your project folder
            self.filename_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.filename_particles_direction = self.relative_path + self.filename_timestamp + "_Particles_Direction.csv"

            # Defining the header row
            header = ['Step', 'Protons_E', 'Protons_SE', 'Protons_SW', 'Protons_W', 'Protons_NW', 'Protons_NE', 'Protons_Rest', 'Neutrons_E', 'Neutrons_SE', 'Neutrons_SW', 'Neutrons_W', 'Neutrons_NW', 'Neutrons_NE', 'Neutrons_Rest', 'Photons_E', 'Photons_SE', 'Photons_SW', 'Photons_W', 'Photons_NW', 'Photons_NE'] 

            with open(self.filename_particles_direction, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(header)

    def step(self):
        """
            Steps the automaton state by one iteration.
            The order of execution of functions is defined in self.execution_order
        """

        #TODO produce an error if function name is not defined in the class 
        for function_name in self.execution_order:
            function = getattr(self, function_name)
            if function_name == 'count_particles' and self.steps_cnt % self.particle_counting_nsteps == 0:
                function()
            else:
                function()
        
        self._worldmap = np.zeros_like(self._worldmap) #(W,H,3)
        self.neutron = np.where(self.particles == 1, 1, 0)
        self.proton = np.where(self.particles == -1, 1, 0)
        self._worldmap[:,:,2]=(self.neutron.sum(axis = 0)/1.)[:,:]
        self._worldmap[:,:,0]=(self.proton.sum(axis = 0)/1.)[:,:]
        if self.constants["photon_visualization"]:
            self.green = np.where (self.photons > 0, 1, 0)
            self._worldmap[:,:,1] = (self.green.sum(axis =0)/1.)[:,:]

        self.steps_cnt += 1

        
    def propagation_prot_neut_step(self):
        """
            Does the propagation step of the protons and neutrons in the automaton
        """
        self.particles = propagation_prot_neut_cpu(self.particles,self.w,self.h,self.dir)

    def propagation_photon_step(self):
        """
            Does the propagation step of the photons in the automaton
        """
        self.photons = propagation_photon_cpu(self.photons,self.w,self.h,self.dir)

    def propagation_anti_photon_step(self):
        """
            Does the propagation step of the photons in the automaton in reverse
            (momentum remains the same but the movement direction is the opposite)
        """
        self.photons = propagation_anti_photon_cpu(self.photons,self.w,self.h,self.dir)

    def sticking_step(self):
        """
        Does the sticking step of the automaton
        """
        #creating a numpy array for constants to give them to sticking_cpu that is using Numba
        sticking_constants = np.array([self.constants["Probability_of_sticking"]])
        (self.particles,self.photons) = sticking_cpu(self.particles,self.photons,self.w,self.h,self.photon_creations['sticking_photon'],self.constants['sticking_prefers_moving_direction'], sticking_constants)
    
    def scattering_step(self):
        """
        Does the scattering step of the automaton
        """
        #creating a numpy array for constants to give them to sticking_cpu that is using Numba
        scattering_constants = np.array([self.constants["Scattering_weight1"], self.constants["Scattering_weight2"], self.constants["Probability of scattering"], self.constants["Scattering_threshold_one"], self.constants["Scattering_threshold_two"]])
        self.particles = scattering_cpu(self.particles,self.w,self.h,scattering_constants)

    def protonaction_step(self):
        """
        Does the proton excusive attribute 
        """
        #creating a numpy array for constants to give them to protonaction_cpu that is using Numba
        protonaction_constants = np.array([self.constants["Prot_Neut_weight1"] , self.constants["Prot_Neut_weight2"] , self.constants["Prot_Neut_threshold"] , self.constants["Prot_Neut_slope"] ])
        (self.particles,self.photons) = protonaction_cpu(self.particles,self.photons,self.w,self.h,self.photon_creations['protonaction_photon'],protonaction_constants)
        
    def neutronaction_step(self):
        """
        Does the neutron exclusive attribute
        """
        #creating a numpy array for constants to give them to neutronaction_cpu that is using Numba
        neutronaction_constants = np.array([self.constants["Prot_Neut_weight1"] , self.constants["Prot_Neut_weight2"] , self.constants["Prot_Neut_threshold"] , self.constants["Prot_Neut_slope"] ])
        (self.particles,self.photons) = neutronaction_cpu(self.particles,self.photons,self.w,self.h,self.photon_creations['neutronaction_photon'],neutronaction_constants)

    def absorption_step(self):
        """
        Does the absorption of photons
        """
        #creating a numpy array for constants to give them to absorption_cpu that is using Numba
        absoprtion_constants = np.array([self.constants["Photon_absorption_probability_0"], self.constants["Photon_absorption_probability_1"], \
                                        self.constants["Photon_absorption_probability_2"], self.constants["Photon_absorption_probability_3"], \
                                        self.constants["Photon_absorption_probability_4"], self.constants["Photon_absorption_probability_5"], \
                                        self.constants["Photon_absorption_probability_6"]])
        (self.particles,self.photons) = absorption_cpu(self.particles,self.photons,self.w,self.h,absoprtion_constants)
        
    def photon_annihilation_step(self):
        
        """
            Annihilates the opposing photons in each direction
        """
        self.photons = photon_annihilation_cpu(self.photons, self.w, self.h)

    def count_particles(self):
        """
            Gives you the statistics of the directions of the particles.
        """
        sum_proton_direction = np.sum(self.particles == -1 , axis=(1,2))
        sum_neutron_direction = np.sum(self.particles == 1 , axis=(1,2))
        sum_photons_direction = np.sum(self.photons , axis=(1,2))
        print("="*80)
        print("Step # = " + str(self.steps_cnt))
        print("Directions: E, SE, SW, W, NW, NE, Rest (Except for photons that cannot be at rest)")
        print("Protons:")
        print(sum_proton_direction)
        print("Neutrons:")
        print(sum_neutron_direction)
        print("Photons:")
        print(sum_photons_direction)
        data = [self.steps_cnt]
        data.extend(sum_proton_direction)
        data.extend(sum_neutron_direction)
        data.extend(sum_photons_direction)

        with open(self.filename_particles_direction, 'a', encoding='UTF8', newline='') as f:
            csv.writer(f).writerow(data)

    def sink_step(self):
        """
            Annihilates the photons in an area called sink
        """
        tmp =[0.0,0.04,0.08,0.04]
        self.photons = sink_cpu(self.photons, self.w, self.h,tmp)

    def source_step(self):
        """
            Creates photons in an area or line or point called source
        """
        self.photons = source_cpu(self.photons, self.w, self.h)
    def arbitrary_step(self):
        """
            This method is supposed to be used for testing and temporary purposes
        """
        (self.particles,self.photons) = arbitrary_cpu(self.particles,self.photons, self.w, self.h)


@njit(parallel=True,cache=True)
def sticking_cpu(particles :np.ndarray ,photons :np.ndarray ,w,h, create_photon, preference, constants):

    absparticles = np.abs(particles)
    newparticles = np.copy(particles)
    total_particles = absparticles.sum(axis=0) #(W,H)

    for x in prange(w):
        for y in prange(h):
            
            if (total_particles[x,y] == 1):
                
                directions_frequency = neighbors_directions(absparticles, x,y, w,h)

                #finding out which direction is dominant, if any
                direction_frequency_maximum = int(directions_frequency.max())

                if (direction_frequency_maximum > 0):

                    dominant_directions = np.where(directions_frequency == direction_frequency_maximum)[0]
                    
                    if np.random.uniform() <= constants[0]:

                        for previousdir in range(7):
                            if (absparticles[previousdir,x,y] == 1):
                                
                                if preference:
                                    if previousdir in dominant_directions:
                                        newdir = previousdir
                                    else:
                                        newdir = np.random.choice(dominant_directions)
                                else:
                                    newdir = np.random.choice(dominant_directions)

                                newparticles[previousdir,x,y] = 0
                                newparticles[newdir,x,y] = particles[previousdir,x,y]
                                #creating photon(s) to conserve momentum
                                if (create_photon):
                                    photons[:,x,y] += photon_creation(previousdir, newdir)
                                
                                break
                                          
    
    return (newparticles,photons)

@njit(parallel = True,cache=True)
def scattering_cpu(particles: np.ndarray, w,h, constants):

    absparticles = np.abs(particles)
    newparticles = np.copy(particles)
    total_particles = absparticles.sum(axis=0) #(W,H)

    for x in prange(w):
        for y in prange(h):

            if (total_particles[x,y] == 2):
                
                nonzero_indices = np.nonzero(particles[:,x,y])[0]

                probability = scattering_probability(particles, w,h, x,y, nonzero_indices[0] ,constants) * \
                    scattering_probability(particles, w,h, x,y, nonzero_indices[1], constants)

                if(np.random.random() < probability):
                    newparticles[:,x,y] = scattering_dynamics_2_particle(particles[:,x,y], nonzero_indices)

            
            elif (total_particles[x,y] == 3):

                nonzero_indices = np.nonzero(particles[:,x,y])[0]

                probability = scattering_probability(particles, w,h, x,y, nonzero_indices[0] ,constants) * \
                    scattering_probability(particles, w,h, x,y, nonzero_indices[1], constants) * \
                    scattering_probability(particles, w,h, x,y, nonzero_indices[2], constants)
                
                if(np.random.random() < probability):
                    newparticles[:,x,y] = scattering_dynamics_3_particle(particles[:,x,y], nonzero_indices)
                


            
    return newparticles

@njit(parallel = True,cache=True)
def protonaction_cpu(particles :np.ndarray ,photons :np.ndarray ,w,h, create_photon, constants):

    newparticles = np.copy(particles)
    absparticles = np.abs(particles)
    partictot = absparticles.sum(axis = 0) #(W,H)
    proton = particles == -1
    protontot = proton.sum(axis = 0)
    for x in prange(w):
        for y in prange(h):
            for state in range(7):
                if particles[state,x,y] == -1 and partictot[x,y] == 1:
                    yplus1 = (y+1)%h
                    xplus1 = (x+1)%w
                    yplus2 = (y+2)%h
                    xplus2 = (x+2)%w
                    xplus3 = (x+3)%w
                    xplus4 = (x+4)%w
                    # The number of protons in the first neighbour (6 neighbours) of this proton 
                    p1 = protontot[xplus2,y] + protontot[xplus1,yplus1] + protontot[x-1,yplus1] + protontot[x-2,y] + protontot[x-1,y-1] + protontot[xplus1,y-1]
                    # The number of protons in the second neighbour (12 neighbours) of this proton 
                    p2 = protontot[xplus4,y] + protontot[xplus3,yplus1] + protontot[xplus2,yplus2] + protontot[x,yplus2] + protontot[x-2,yplus2] + protontot[x-3,yplus1] \
                    + protontot[x-4,y] + protontot[x-3,y-1] + protontot[x-2,y-2] + protontot[x,y-2] + protontot[xplus2,y-2] + protontot[xplus3,y-1]
                    p_eff = constants[0] * p1 + constants[1] * p2
                    if np.random.uniform() < (p_eff - constants[2])* constants[3]:
                        previousdir = state
                        newdir = np.random.choice(np.arange(6))
                        newparticles[previousdir,x,y] = 0
                        newparticles[newdir,x,y] = -1

                        #creating photon(s) to conserve momentum
                        if (create_photon):
                            photons[:,x,y] += photon_creation(previousdir, newdir)
            
    return (newparticles,photons)

@njit(parallel = True,cache=True)
def neutronaction_cpu(particles :np.ndarray ,photons :np.ndarray ,w,h, create_photon, constants):

    newparticles = np.copy(particles)
    absparticles = np.abs(particles)
    partictot = absparticles.sum(axis = 0) #(W,H)
    neutron = particles == 1
    neutrontot = neutron.sum(axis = 0)
    for x in prange(w):
        for y in prange(h):
            for state in range(7):
                if particles[state,x,y] == 1 and partictot[x,y] == 1:
                    yplus1 = (y+1)%h
                    xplus1 = (x+1)%w
                    yplus2 = (y+2)%h
                    xplus2 = (x+2)%w
                    xplus3 = (x+3)%w
                    xplus4 = (x+4)%w
                    # The number of neutrons in the first neighbour (6 neighbours) of this neutron 
                    n1 = neutrontot[xplus2,y] + neutrontot[xplus1,yplus1] + neutrontot[x-1,yplus1] + neutrontot[x-2,y] + neutrontot[x-1,y-1] + neutrontot[xplus1,y-1]
                    # The number of neutrons in the second neighbour (12 neighbours) of this neutron 
                    n2 = neutrontot[xplus4,y] + neutrontot[xplus3,yplus1] + neutrontot[xplus2,yplus2] + neutrontot[x,yplus2] + neutrontot[x-2,yplus2] + neutrontot[x-3,yplus1] \
                    + neutrontot[x-4,y] + neutrontot[x-3,y-1] + neutrontot[x-2,y-2] + neutrontot[x,y-2] + neutrontot[xplus2,y-2] + neutrontot[xplus3,y-1]
                    n_eff = constants[0] * n1 + constants[1] * n2
                    if np.random.uniform() < (n_eff - constants[2]) * constants[3]:
                        previousdir = state
                        newdir = np.random.choice(np.arange(6))
                        newparticles[previousdir,x,y] = 0
                        newparticles[newdir,x,y] = 1

                        #creating photon(s) to conserve momentum
                        if (create_photon):
                            photons[:,x,y] += photon_creation(previousdir, newdir)
                            
    return (newparticles,photons)

@njit(parallel=True,cache=True)
def propagation_prot_neut_cpu(particles, w,h,dirdico):
    newparticles=np.copy(particles)
    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            for dir in range(6):
                new_position = (loc+dirdico[dir])%np.array([w,h])
                newparticles[dir,new_position[0],new_position[1]] = particles[dir,x,y]
    return newparticles

@njit(parallel=True,cache=True)
def propagation_photon_cpu(photons, w,h,dirdico):
    newphotons = np.copy(photons)
    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            for dir in range(6):
                new_position = (loc+dirdico[dir])%np.array([w,h])
                newphotons[dir,new_position[0],new_position[1]] = photons[dir,x,y]
    return newphotons

@njit(parallel=True,cache=True)
def propagation_anti_photon_cpu(photons, w,h,dirdico):
    newphotons = np.copy(photons)
    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            for dir in range(6):
                new_position = (loc-dirdico[dir])%np.array([w,h])
                newphotons[dir,new_position[0],new_position[1]] = photons[dir,x,y]
    return newphotons

@njit(parallel=True,cache=True)
def absorption_cpu(particles: np.ndarray, photons: np.ndarray, w,h, constants):

    newparticles = np.copy(particles)
    absparticles = np.abs(particles)

    for x in prange(w):
        for y in prange(h):
            
            #choosing directions of photons to consider in order randomly
            photons_directions_shuffle = np.nonzero(photons[:,x,y])[0]
            np.random.shuffle(photons_directions_shuffle)

            for photon_direction in photons_directions_shuffle:
                for photon_number in range(photons[photon_direction,x,y]):

                    #shuffling particles each time we are considering a photon
                    particles_directions_shuffle = np.nonzero(newparticles[:,x,y])[0]
                    np.random.shuffle(particles_directions_shuffle)

                    for particle_direction in particles_directions_shuffle:

                        same_direction_moving_neighbors_number = neighbors_directions(absparticles, x,y, w,h)[particle_direction]

                        probability = constants[int(same_direction_moving_neighbors_number)]

                        if(np.random.uniform() < probability):

                            #absorption commands
                            if(particle_direction == 6):
                                if(newparticles[photon_direction,x,y] == 0):
                                    new_particle_direction = photon_direction
                                    #! Warning: We used newparticles not particles because absorption of previous photons affects next absorptions
                                    #! This dependancy is local, so it is okay that we probe nodes in the lattice
                                    newparticles[new_particle_direction,x,y] = newparticles[particle_direction,x,y]
                                    newparticles[particle_direction,x,y] = 0

                                    photons[photon_direction,x,y] -= 1
                                    break
                            
                            else:

                                momentum_difference = (photon_direction - particle_direction) % 6
                                
                                if(momentum_difference == 3):
                                    new_particle_direction = 6
                                    if(newparticles[new_particle_direction,x,y] == 0):
                                        newparticles[new_particle_direction,x,y] = newparticles[particle_direction,x,y]
                                        newparticles[particle_direction,x,y] = 0

                                        photons[photon_direction,x,y] -= 1
                                        break

                                elif(momentum_difference == 2):
                                    new_particle_direction = (1 + particle_direction) % 6
                                    if(newparticles[new_particle_direction,x,y] == 0):
                                        newparticles[new_particle_direction,x,y] = newparticles[particle_direction,x,y]
                                        newparticles[particle_direction,x,y] = 0

                                        photons[photon_direction,x,y] -= 1
                                        break
                                
                                elif(momentum_difference == 4):
                                    new_particle_direction = (5 + particle_direction) % 6
                                    if(newparticles[new_particle_direction,x,y] == 0):
                                        newparticles[new_particle_direction,x,y] = newparticles[particle_direction,x,y]
                                        newparticles[particle_direction,x,y] = 0

                                        photons[photon_direction,x,y] -= 1
                                        break
    
    return (newparticles,photons)

@njit(cache=True)
def scattering_dynamics_2_particle(particles, nonzero_indices):
    newparticles = np.copy(particles)

    smaller_index = nonzero_indices[0]
    bigger_index = nonzero_indices[1]
    momentum_difference = bigger_index - smaller_index

    if (bigger_index == 6):
        if (np.random.uniform() < 0.5):
            newparticles[bigger_index] = particles[smaller_index]
            newparticles[smaller_index] = particles[bigger_index]
        else:
            newparticles[smaller_index] = 0
            newparticles[bigger_index] = 0

            if (np.random.uniform() < 0.5):
                newparticles[(1 + smaller_index) % 6] = particles[smaller_index]
                newparticles[(5 + smaller_index) % 6] = particles[bigger_index]
            else:
                newparticles[(1 + smaller_index) % 6] = particles[bigger_index]
                newparticles[(5 + smaller_index) % 6] = particles[smaller_index]
    
    else:
        if (momentum_difference == 1):
            newparticles[bigger_index] = particles[smaller_index]
            newparticles[smaller_index] = particles[bigger_index]
            
        elif (momentum_difference == 2):
            if (np.random.uniform() < 0.5):
                newparticles[bigger_index] = particles[smaller_index]
                newparticles[smaller_index] = particles[bigger_index]
            else:
                newparticles[smaller_index] = 0
                newparticles[bigger_index] = 0

                if(np.random.uniform() < 0.5):
                    newparticles[(2 + smaller_index) % 6] = particles[smaller_index]
                    newparticles[6] = particles[bigger_index]
                else:
                    newparticles[(2 + smaller_index) % 6] = particles[bigger_index]
                    newparticles[6] = particles[smaller_index]
        
        elif (momentum_difference == 3):
            rand = np.random.uniform()
            if (rand < 1/3):
                newparticles[bigger_index] = particles[smaller_index]
                newparticles[smaller_index] = particles[bigger_index]

            elif (rand>1/3 and rand<2/3):
                newparticles[smaller_index] = 0
                newparticles[bigger_index] = 0

                if(np.random.uniform() < 0.5):
                    newparticles[(1 + smaller_index) % 6] = particles[smaller_index]
                    newparticles[(4 + smaller_index) % 6] = particles[bigger_index]
                else:
                    newparticles[(1 + smaller_index) % 6] = particles[bigger_index]
                    newparticles[(4 + smaller_index) % 6] = particles[smaller_index]
            
            else:
                newparticles[smaller_index] = 0
                newparticles[bigger_index] = 0

                if(np.random.uniform() < 0.5):
                    newparticles[(2 + smaller_index) % 6] = particles[smaller_index]
                    newparticles[(5 + smaller_index) % 6] = particles[bigger_index]
                else:
                    newparticles[(2 + smaller_index) % 6] = particles[bigger_index]
                    newparticles[(5 + smaller_index) % 6] = particles[smaller_index]

        elif (momentum_difference == 4):
            if (np.random.uniform() < 0.5):
                newparticles[bigger_index] = particles[smaller_index]
                newparticles[smaller_index] = particles[bigger_index]
            else:
                newparticles[smaller_index] = 0
                newparticles[bigger_index] = 0

                if(np.random.uniform() < 0.5):
                    newparticles[(5 + smaller_index) % 6] = particles[smaller_index]
                    newparticles[6] = particles[bigger_index]
                else:
                    newparticles[(5 + smaller_index) % 6] = particles[bigger_index]
                    newparticles[6] = particles[smaller_index]

        elif (momentum_difference == 5):
            newparticles[bigger_index] = particles[smaller_index]
            newparticles[smaller_index] = particles[bigger_index]
    
    return newparticles

@njit(cache=True)
def scattering_dynamics_3_particle(particles, nonzero_indices):
    newparticles = np.copy(particles)

    index_1 = nonzero_indices[0]
    index_2 = nonzero_indices[1]
    index_3 = nonzero_indices[2]

    momentum_diff_12 = index_2 - index_1
    momentum_diff_23 = index_3 - index_2

    nonzero_values = particles[nonzero_indices]
    count_protons = np.count_nonzero(nonzero_values == -1)
    count_neutrons = np.count_nonzero(nonzero_values == 1)
    if (count_protons == 1):
        single_particle = -1
    elif (count_neutrons == 1):
        single_particle = 1
    else:
        single_particle = None

    if (index_3 == 6):
        new_index_1 = 0
        new_index_2 = momentum_diff_12
        new_momentum_diff_12 = new_index_2 - new_index_1

        if (new_momentum_diff_12 == 1):
            #TODO exclude the non-scattered permutation
            if (single_particle is not None):
                #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                newparticles[nonzero_indices] = 0
                rand = np.random.uniform()
                if (rand < 1/3):
                    newparticles[(new_index_1 + index_1) % 6] = single_particle
                    newparticles[(new_index_2 + index_1) % 6] = -single_particle
                    newparticles[6] = -single_particle
                elif (rand > 1/3 and rand < 2/3):
                    newparticles[(new_index_1 + index_1) % 6] = -single_particle
                    newparticles[(new_index_2 + index_1) % 6] = single_particle
                    newparticles[6] = -single_particle
                else:
                    newparticles[(new_index_1 + index_1) % 6] = -single_particle
                    newparticles[(new_index_2 + index_1) % 6] = -single_particle
                    newparticles[6] = single_particle
                 
        elif (new_momentum_diff_12 == 2):
            
            rand1 = np.random.uniform()
            if (rand1 < 1/3):
                if (single_particle is not None):
                    newparticles[nonzero_indices] = 0
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(new_index_1 + index_1) % 6] = single_particle
                        newparticles[(new_index_2 + index_1) % 6] = -single_particle
                        newparticles[6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(new_index_1 + index_1) % 6] = -single_particle
                        newparticles[(new_index_2 + index_1) % 6] = single_particle
                        newparticles[6] = -single_particle
                    else:
                        newparticles[(new_index_1 + index_1) % 6] = -single_particle
                        newparticles[(new_index_2 + index_1) % 6] = -single_particle
                        newparticles[6] = single_particle
                    
            elif (rand1 > 1/3 and rand1 < 2/3):
                newparticles[nonzero_indices] = 0
                if (single_particle is None):
                    newparticles[(0 + index_1) % 6] = particles[index_1]
                    newparticles[(1 + index_1) % 6] = particles[index_1]
                    newparticles[(3 + index_1) % 6] = particles[index_1]
                else:
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(0 + index_1) % 6] = single_particle
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(3 + index_1) % 6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(0 + index_1) % 6] = -single_particle
                        newparticles[(1 + index_1) % 6] = single_particle
                        newparticles[(3 + index_1) % 6] = -single_particle
                    else:
                        newparticles[(0 + index_1) % 6] = -single_particle
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(3 + index_1) % 6] = single_particle

            else:
                newparticles[nonzero_indices] = 0
                if (single_particle is None):
                    newparticles[(1 + index_1) % 6] = particles[index_1]
                    newparticles[(2 + index_1) % 6] = particles[index_1]
                    newparticles[(5 + index_1) % 6] = particles[index_1]
                else:
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(1 + index_1) % 6] = single_particle
                        newparticles[(2 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(2 + index_1) % 6] = single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                    else:
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(2 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = single_particle
        
        elif (new_momentum_diff_12 == 3):
            rand1 = np.random.uniform()
            if (rand1 < 0.2):
                if (single_particle is not None):
                    #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                    newparticles[nonzero_indices] = 0
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(new_index_1 + index_1) % 6] = single_particle
                        newparticles[(new_index_2 + index_1) % 6] = -single_particle
                        newparticles[6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(new_index_1 + index_1) % 6] = -single_particle
                        newparticles[(new_index_2 + index_1) % 6] = single_particle
                        newparticles[6] = -single_particle
                    else:
                        newparticles[(new_index_1 + index_1) % 6] = -single_particle
                        newparticles[(new_index_2 + index_1) % 6] = -single_particle
                        newparticles[6] = single_particle

            elif (rand1 > 0.2 and rand1 < 0.4):
                newparticles[nonzero_indices] = 0
                if (single_particle is None):
                    newparticles[(1 + index_1) % 6] = particles[index_1]
                    newparticles[(4 + index_1) % 6] = particles[index_1]
                    newparticles[6] = particles[index_1]
                else:
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(1 + index_1) % 6] = single_particle
                        newparticles[(4 + index_1) % 6] = -single_particle
                        newparticles[6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(4 + index_1) % 6] = single_particle
                        newparticles[6] = -single_particle
                    else:
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(4 + index_1) % 6] = -single_particle
                        newparticles[6] = single_particle
                
            elif (rand1 > 0.4 and rand1 < 0.6):
                newparticles[nonzero_indices] = 0
                if (single_particle is None):
                    newparticles[(2 + index_1) % 6] = particles[index_1]
                    newparticles[(5 + index_1) % 6] = particles[index_1]
                    newparticles[6] = particles[index_1]
                else:
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(2 + index_1) % 6] = single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                        newparticles[6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(2 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = single_particle
                        newparticles[6] = -single_particle
                    else:
                        newparticles[(2 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                        newparticles[6] = single_particle

            elif (rand1 > 0.6 and rand1 < 0.8):
                newparticles[nonzero_indices] = 0
                if (single_particle is None):
                    newparticles[(0 + index_1) % 6] = particles[index_1]
                    newparticles[(2 + index_1) % 6] = particles[index_1]
                    newparticles[(4 + index_1) % 6] = particles[index_1]
                else:
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(0 + index_1) % 6] = single_particle
                        newparticles[(2 + index_1) % 6] = -single_particle
                        newparticles[(4 + index_1) % 6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(0 + index_1) % 6] = -single_particle
                        newparticles[(2 + index_1) % 6] = single_particle
                        newparticles[(4 + index_1) % 6] = -single_particle
                    else:
                        newparticles[(0 + index_1) % 6] = -single_particle
                        newparticles[(2 + index_1) % 6] = -single_particle
                        newparticles[(4 + index_1) % 6] = single_particle

            else:
                newparticles[nonzero_indices] = 0
                if (single_particle is None):
                    newparticles[(1 + index_1) % 6] = particles[index_1]
                    newparticles[(3 + index_1) % 6] = particles[index_1]
                    newparticles[(5 + index_1) % 6] = particles[index_1]
                else:
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(1 + index_1) % 6] = single_particle
                        newparticles[(3 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(3 + index_1) % 6] = single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                    else:
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(3 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = single_particle

        elif (new_momentum_diff_12 == 4):
            rand1 = np.random.uniform()
            if (rand1 < 1/3):
                if (single_particle is not None):
                    #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                    newparticles[nonzero_indices] = 0
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(new_index_1 + index_1) % 6] = single_particle
                        newparticles[(new_index_2 + index_1) % 6] = -single_particle
                        newparticles[6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(new_index_1 + index_1) % 6] = -single_particle
                        newparticles[(new_index_2 + index_1) % 6] = single_particle
                        newparticles[6] = -single_particle
                    else:
                        newparticles[(new_index_1 + index_1) % 6] = -single_particle
                        newparticles[(new_index_2 + index_1) % 6] = -single_particle
                        newparticles[6] = single_particle
                    
            elif (rand1 > 1/3 and rand1 < 2/3):
                newparticles[nonzero_indices] = 0
                if (single_particle is None):
                    newparticles[(0 + index_1) % 6] = particles[index_1]
                    newparticles[(3 + index_1) % 6] = particles[index_1]
                    newparticles[(5 + index_1) % 6] = particles[index_1]
                else:
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(0 + index_1) % 6] = single_particle
                        newparticles[(3 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(0 + index_1) % 6] = -single_particle
                        newparticles[(3 + index_1) % 6] = single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                    else:
                        newparticles[(0 + index_1) % 6] = -single_particle
                        newparticles[(3 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = single_particle

            else:
                newparticles[nonzero_indices] = 0
                if (single_particle is None):
                    newparticles[(1 + index_1) % 6] = particles[index_1]
                    newparticles[(4 + index_1) % 6] = particles[index_1]
                    newparticles[(5 + index_1) % 6] = particles[index_1]
                else:
                    rand2 = np.random.uniform()
                    if (rand2 < 1/3):
                        newparticles[(1 + index_1) % 6] = single_particle
                        newparticles[(4 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                    elif (rand2 > 1/3 and rand2 < 2/3):
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(4 + index_1) % 6] = single_particle
                        newparticles[(5 + index_1) % 6] = -single_particle
                    else:
                        newparticles[(1 + index_1) % 6] = -single_particle
                        newparticles[(4 + index_1) % 6] = -single_particle
                        newparticles[(5 + index_1) % 6] = single_particle

        elif (new_momentum_diff_12 == 5):
            #TODO exclude the non-scattered permutation
            if (single_particle is not None):
                #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                newparticles[nonzero_indices] = 0
                rand = np.random.uniform()
                if (rand < 1/3):
                    newparticles[(new_index_1 + index_1) % 6] = single_particle
                    newparticles[(new_index_2 + index_1) % 6] = -single_particle
                    newparticles[6] = -single_particle
                elif (rand > 1/3 and rand < 2/3):
                    newparticles[(new_index_1 + index_1) % 6] = -single_particle
                    newparticles[(new_index_2 + index_1) % 6] = single_particle
                    newparticles[6] = -single_particle
                else:
                    newparticles[(new_index_1 + index_1) % 6] = -single_particle
                    newparticles[(new_index_2 + index_1) % 6] = -single_particle
                    newparticles[6] = single_particle

    else:
        if (momentum_diff_12 == 1):
            if (momentum_diff_23 == 1):
                #TODO exclude the non-scattered permutation
                if (single_particle is not None):
                    #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                    newparticles[nonzero_indices] = 0
                    rand = np.random.uniform()
                    if (rand < 1/3):
                        newparticles[index_1] = single_particle
                        newparticles[index_2] = -single_particle
                        newparticles[index_3] = -single_particle
                    elif (rand > 1/3 and rand < 2/3):
                        newparticles[index_1] = -single_particle
                        newparticles[index_2] = single_particle
                        newparticles[index_3] = -single_particle
                    else:
                        newparticles[index_1] = -single_particle
                        newparticles[index_2] = -single_particle
                        newparticles[index_3] = single_particle
            
            elif (momentum_diff_23 == 2):
                rand1 = np.random.uniform()
                if (rand1 < 1/3):
                    if (single_particle is not None):
                        #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                        newparticles[nonzero_indices] = 0
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[index_1] = single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = single_particle
                            newparticles[index_3] = -single_particle
                        else:
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = single_particle
                        
                elif (rand1 > 1/3 and rand1 < 2/3):
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(1 + index_1) % 6] = particles[index_1]
                        newparticles[(2 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(1 + index_1) % 6] = single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        else:
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle

                else:
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(0 + index_1) % 6] = particles[index_1]
                        newparticles[(2 + index_1) % 6] = particles[index_1]
                        newparticles[6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(0 + index_1) % 6] = single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = single_particle
                            newparticles[6] = -single_particle
                        else:
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[6] = single_particle
            
            elif (momentum_diff_23 == 3):
                
                rand1 = np.random.uniform()
                if (rand1 < 1/3):
                    if (single_particle is not None):
                        #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                        newparticles[nonzero_indices] = 0
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[index_1] = single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = single_particle
                            newparticles[index_3] = -single_particle
                        else:
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = single_particle
                        
                elif (rand1 > 1/3 and rand1 < 2/3):
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(0 + index_1) % 6] = particles[index_1]
                        newparticles[(2 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(0 + index_1) % 6] = single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        else:
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle

                else:
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(1 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                        newparticles[6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(1 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                            newparticles[6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle
                            newparticles[6] = -single_particle
                        else:
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                            newparticles[6] = single_particle

            elif (momentum_diff_23 == 4):
                #TODO exclude the non-scattered permutation
                if (single_particle is not None):
                    #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                    newparticles[nonzero_indices] = 0
                    rand = np.random.uniform()
                    if (rand < 1/3):
                        newparticles[index_1] = single_particle
                        newparticles[index_2] = -single_particle
                        newparticles[index_3] = -single_particle
                    elif (rand > 1/3 and rand < 2/3):
                        newparticles[index_1] = -single_particle
                        newparticles[index_2] = single_particle
                        newparticles[index_3] = -single_particle
                    else:
                        newparticles[index_1] = -single_particle
                        newparticles[index_2] = -single_particle
                        newparticles[index_3] = single_particle
        
        elif (momentum_diff_12 == 2):
            if (momentum_diff_23 == 1):
                rand1 = np.random.uniform()
                if (rand1 < 1/3):
                    if (single_particle is not None):
                        #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                        newparticles[nonzero_indices] = 0
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[index_1] = single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = single_particle
                            newparticles[index_3] = -single_particle
                        else:
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = single_particle
                        
                elif (rand1 > 1/3 and rand1 < 2/3):
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(1 + index_1) % 6] = particles[index_1]
                        newparticles[(2 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(1 + index_1) % 6] = single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        else:
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle

                else:
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(0 + index_1) % 6] = particles[index_1]
                        newparticles[(2 + index_1) % 6] = particles[index_1]
                        newparticles[6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(0 + index_1) % 6] = single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = single_particle
                            newparticles[6] = -single_particle
                        else:
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[6] = single_particle

            elif (momentum_diff_23 == 2):
                rand1 = np.random.uniform()
                if (rand1 < 0.2):
                    if (single_particle is not None):
                        #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                        newparticles[nonzero_indices] = 0
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[index_1] = single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = single_particle
                            newparticles[index_3] = -single_particle
                        else:
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = single_particle
                        
                elif (rand1 > 0.2 and rand1 < 0.4):
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(1 + index_1) % 6] = particles[index_1]
                        newparticles[(3 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(1 + index_1) % 6] = single_particle
                            newparticles[(3 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(3 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        else:
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(3 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle

                elif (rand1 > 0.4 and rand1 < 0.6):
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(0 + index_1) % 6] = particles[index_1]
                        newparticles[(3 + index_1) % 6] = particles[index_1]
                        newparticles[6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(0 + index_1) % 6] = single_particle
                            newparticles[(3 + index_1) % 6] = -single_particle
                            newparticles[6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(3 + index_1) % 6] = single_particle
                            newparticles[6] = -single_particle
                        else:
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(3 + index_1) % 6] = -single_particle
                            newparticles[6] = single_particle

                elif (rand1 > 0.6 and rand1 < 0.8):
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(1 + index_1) % 6] = particles[index_1]
                        newparticles[(4 + index_1) % 6] = particles[index_1]
                        newparticles[6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(1 + index_1) % 6] = single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                            newparticles[6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = single_particle
                            newparticles[6] = -single_particle
                        else:
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                            newparticles[6] = single_particle

                else:
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(2 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                        newparticles[6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(2 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                            newparticles[6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle
                            newparticles[6] = -single_particle
                        else:
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                            newparticles[6] = single_particle

            
            elif (momentum_diff_23 == 3):
                rand1 = np.random.uniform()
                if (rand1 < 1/3):
                    if (single_particle is not None):
                        #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                        newparticles[nonzero_indices] = 0
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[index_1] = single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = single_particle
                            newparticles[index_3] = -single_particle
                        else:
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = single_particle
                        
                elif (rand1 > 1/3 and rand1 < 2/3):
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(0 + index_1) % 6] = particles[index_1]
                        newparticles[(1 + index_1) % 6] = particles[index_1]
                        newparticles[(4 + index_1) % 6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(0 + index_1) % 6] = single_particle
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(1 + index_1) % 6] = single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                        else:
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = single_particle

                else:
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(1 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                        newparticles[6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(1 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                            newparticles[6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle
                            newparticles[6] = -single_particle
                        else:
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                            newparticles[6] = single_particle
            
        elif (momentum_diff_12 == 3):
            if (momentum_diff_23 == 1):
                rand1 = np.random.uniform()
                if (rand1 < 1/3):
                    if (single_particle is not None):
                        #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                        newparticles[nonzero_indices] = 0
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[index_1] = single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = single_particle
                            newparticles[index_3] = -single_particle
                        else:
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = single_particle
                        
                elif (rand1 > 1/3 and rand1 < 2/3):
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(2 + index_1) % 6] = particles[index_1]
                        newparticles[(4 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(2 + index_1) % 6] = single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        else:
                            newparticles[(2 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle

                else:
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(3 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                        newparticles[6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(3 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                            newparticles[6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(3 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle
                            newparticles[6] = -single_particle
                        else:
                            newparticles[(3 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                            newparticles[6] = single_particle
            
            elif (momentum_diff_23 == 2):
                rand1 = np.random.uniform()
                if (rand1 < 1/3):
                    if (single_particle is not None):
                        #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                        newparticles[nonzero_indices] = 0
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[index_1] = single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = single_particle
                            newparticles[index_3] = -single_particle
                        else:
                            newparticles[index_1] = -single_particle
                            newparticles[index_2] = -single_particle
                            newparticles[index_3] = single_particle
                        
                elif (rand1 > 1/3 and rand1 < 2/3):
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(1 + index_1) % 6] = particles[index_1]
                        newparticles[(4 + index_1) % 6] = particles[index_1]
                        newparticles[(5 + index_1) % 6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(1 + index_1) % 6] = single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = single_particle
                            newparticles[(5 + index_1) % 6] = -single_particle
                        else:
                            newparticles[(1 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                            newparticles[(5 + index_1) % 6] = single_particle

                else:
                    newparticles[nonzero_indices] = 0
                    if (single_particle is None):
                        newparticles[(0 + index_1) % 6] = particles[index_1]
                        newparticles[(4 + index_1) % 6] = particles[index_1]
                        newparticles[6] = particles[index_1]
                    else:
                        rand2 = np.random.uniform()
                        if (rand2 < 1/3):
                            newparticles[(0 + index_1) % 6] = single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                            newparticles[6] = -single_particle
                        elif (rand2 > 1/3 and rand2 < 2/3):
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = single_particle
                            newparticles[6] = -single_particle
                        else:
                            newparticles[(0 + index_1) % 6] = -single_particle
                            newparticles[(4 + index_1) % 6] = -single_particle
                            newparticles[6] = single_particle
        
        elif (momentum_diff_12 == 4):
            if (momentum_diff_23 == 1):
                #TODO exclude the non-scattered permutation
                if (single_particle is not None):
                    #! You set newparticles = 0 for all cases. If you want to exclude a case, check it.
                    newparticles[nonzero_indices] = 0
                    rand = np.random.uniform()
                    if (rand < 1/3):
                        newparticles[index_1] = single_particle
                        newparticles[index_2] = -single_particle
                        newparticles[index_3] = -single_particle
                    elif (rand > 1/3 and rand < 2/3):
                        newparticles[index_1] = -single_particle
                        newparticles[index_2] = single_particle
                        newparticles[index_3] = -single_particle
                    else:
                        newparticles[index_1] = -single_particle
                        newparticles[index_2] = -single_particle
                        newparticles[index_3] = single_particle

    return newparticles

@njit(cache=True)
def scattering_probability(particles, w,h, x,y, dir, constants):

    xplus1 = (x+1)%w
    yplus1 = (y+1)%h
    yplus2 = (y+2)%h
    xplus2 = (x+2)%w
    xplus3 = (x+3)%w
    xplus4 = (x+4)%w
    base_probability = constants[2]
    threshold_one = constants[3]
    threshold_two = constants[4]

    first_neighbors =  particles[dir, xplus2, y] + particles[dir, xplus1, yplus1] + particles[dir, x-1, yplus1] +  \
        particles [dir, x-2, y] + particles[dir, x-1, y-1] + particles[dir, xplus1, y-1]
    
    second_neighbors = particles[dir, xplus4, y] + particles[dir, xplus3, yplus1] + particles[dir, xplus2, yplus2] + \
        particles[dir, x, yplus2] + particles[dir, x-2, yplus2] + particles[dir, x-3, yplus1] + \
        particles[dir, x-4, y] + particles[dir, x-3, y-1] + particles[dir, x-2, y-2] + particles[dir, x, y-2] + \
        particles[dir, xplus2, y-2] + particles[dir, xplus3, y-1]
    
    total_neighbors = constants[0] * first_neighbors + constants[1] * second_neighbors

    probability = base_probability
    if (total_neighbors > threshold_one):
        probability = np.maximum(0, base_probability * (threshold_two - total_neighbors) / (threshold_two - threshold_one))

    return probability

@njit(cache=True)
def photon_creation(previousdir, newdir):
    newphotons = np.zeros(6,dtype=np.int64)
    if (previousdir != newdir):
        if(previousdir == 6):
            newphotons[(newdir + 3) % 6] += 1

        elif(newdir == 6):
            newphotons[previousdir] += 1
                            
        else:

            momentum_difference = (newdir - previousdir) % 6
                                    
            if(momentum_difference == 1):
                newphotons[(5 + previousdir) % 6] += 1
                                    
            elif(momentum_difference == 2):
                newphotons[previousdir] += 1
                newphotons[(5 + previousdir) % 6] += 1
                                    
            elif(momentum_difference == 3):
                newphotons[previousdir] += 2
                                    
            elif(momentum_difference == 4):
                newphotons[previousdir] += 1
                newphotons[(1 + previousdir) % 6] += 1
                                    
            elif(momentum_difference == 5):
                newphotons[(1 + previousdir) % 6] += 1
    
    return newphotons

@njit(cache=True)
def neighbors_directions(absparticles, x,y, w,h):

    #This function gives a numpy array of size 7 in which number of neighbors moving in each direction is stored. (Starting from East to Rest)
    yplus1 = (y+1)%h
    xplus1 = (x+1)%w
    xplus2 = (x+2)%w
    
    result = np.zeros(7, dtype=absparticles.dtype)
    result += absparticles[:, xplus2, y]
    result += absparticles[:, xplus1, yplus1]
    result += absparticles[:, x - 1, yplus1]
    result += absparticles[:, x - 2, y]
    result += absparticles[:, x - 1, y - 1]
    result += absparticles[:, xplus1, y - 1]
        
    return result
           
@njit(parallel = True, cache = True)
def photon_annihilation_cpu(photons, w,h):

    for x in prange(w):
        for y in prange(h):

            Net_photons_E = photons[0,x,y] - photons[3,x,y]
            Net_photons_SE = photons[1,x,y] - photons[4,x,y]
            Net_photons_SW = photons[2,x,y] - photons[5,x,y]

            if (Net_photons_E >= 0):
                photons[0,x,y] = Net_photons_E
                photons[3,x,y] = 0
            else:
                photons[0,x,y] = 0
                photons[3,x,y] = - Net_photons_E

            if (Net_photons_SE >= 0):
                photons[1,x,y] = Net_photons_SE
                photons[4,x,y] = 0
            else:
                photons[1,x,y] = 0
                photons[4,x,y] = - Net_photons_SE

            if (Net_photons_SW >= 0):
                photons[2,x,y] = Net_photons_SW
                photons[5,x,y] = 0
            else:
                photons[2,x,y] = 0
                photons[5,x,y] = - Net_photons_SW

    return photons

#! When you are creating a new sink or source don't forget that the our lattice is squared and by ignoring some of its nodes we make it hexagonal.
@njit(parallel = True, cache = True)
def sink_cpu(photons, w,h,sink_value):
    
    for i in prange(6):
        for x in prange (int(4/5*w),int(5/5*w)):
            for y in prange (h):
                if np.random.uniform() < sink_value[0]:
                    photons[i,x,y] = 0
    
    for i in prange(6):
        for x in prange (int(3/5*w),int(4/5*w)):
            for y in prange (h):
                if np.random.uniform() < sink_value[1]:
                    photons[i,x,y] = 0
                    
    for i in prange(6):
        for x in prange (int(2/5*w),int(3/5*w)):
            for y in prange(h):
                if np.random.uniform() < sink_value[2]:
                    photons[i,x,y] = 0
    
    for i in range(6):
        for x in prange (int(1/5*w),int(2/5*w)):
            for y in prange(h):
                if np.random.uniform() < sink_value[3]:
                    photons[i,x,y] = 0
                    
            
                
    

    return photons

@njit(parallel = True, cache = True)
def source_cpu(photons, w,h):
    photons[:,0,1::2] = 1
    photons[:,1,0::2] = 1
                
        
    return photons

@njit(parallel = True, cache = True)
def arbitrary_cpu(particles: np.ndarray, photons: np.ndarray,w,h):
    return (particles, photons)