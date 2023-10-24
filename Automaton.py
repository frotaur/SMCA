import numpy as np
from numba import njit, prange,cuda 
from Config import *
import numba.typed as numt
import cv2
import csv
import datetime
import sys
import concurrent.futures

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

    def __init__(self, size, photon_creation_map, execution_order, constants, Init_particles): # size = (W,H) "Note: W,H must be even numbers." Configuration is a list of booleans. Constants is a dictionary.
        super().__init__(size)
        self.steps_cnt = 0
        # 0,1,2,3,4,5 of the first dimension are the six directions, starting with East, and continuing clockwise
        
        #create a lattice in which there are some neutrons and protons. 0: nothing, -1: proton, 1: neutron
        self.particles = Init_particles

        #creating an array for photons
        self.photons = np.zeros((6,self.w,self.h),dtype=int) #dimensions of this array are the same with particles and the values of it are the number of photons in that location and direction
        
        self.dir = np.array([[2,0],[1,1],[-1,1],[-2,0],[-1,-1],[1,-1]])  # Contains arrays of the six directions, starting with East, and continuing clockwise

        
        self.photon_creation_map = photon_creation_map
        self.photon_creation_config = self.photon_creation_map['config_list']
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

        for function_name in self.execution_order:
            function = getattr(self, function_name)
            if function_name == 'count_particles' and self.steps_cnt % self.particle_counting_nsteps == 0:
                function()
            else:
                function()
        
        self._worldmap = np.zeros_like(self._worldmap) #(W,H,3)
        self.neutron = np.where(self.particles == 1, 1, 0)
        self.proton = np.where(self.particles == -1, 1, 0)
        self.green = np.where (self.photons > 0, 1, 0)
        self._worldmap[:,:,1] = (self.green.sum(axis =0)/1.)[:,:]
        self._worldmap[:,:,2]=(self.neutron.sum(axis = 0)/1.)[:,:]
        self._worldmap[:,:,0]=(self.proton.sum(axis = 0)/1.)[:,:]

        self.steps_cnt += 1
        

        
        
        
    def propagation_step(self):
        """
            Does the propagation step of the automaton
        """
        (self.particles,self.photons) = propagation_cpu(self.particles,self.photons,self.w,self.h,self.dir)

    def collision_step(self):
        """
        Does the collision step of the automaton
        """
        #creating a numpy array for constants to give them to collision_cpu that is using Numba
        collision_constants = np.array([self.constants["Probability_of_sticking"] , self.constants["Sticking_w1_input"] , self.constants["Sticking_w2_input"] , self.constants["Sticking_w3_input"] , self.constants["Sticking_w4_input"], self.constants["Sticking_high_threshold"], self.constants["Sticking_low_threshold"] ])
        (self.particles,self.photons) = collision_cpu(self.particles,self.photons,self.w,self.h,self.photon_creation_config[self.photon_creation_map.get('Collision_photon')],collision_constants)
    
    def protonaction_step(self):
        """
        Does the proton excusive attribute 
        """
        #creating a numpy array for constants to give them to protonaction_cpu that is using Numba
        protonaction_constants = np.array([self.constants["Prot_Neut_weight1"] , self.constants["Prot_Neut_weight2"] , self.constants["Prot_Neut_threshold"] , self.constants["Prot_Neut_slope"] ])
        (self.particles,self.photons) = protonaction_cpu(self.particles,self.photons,self.w,self.h,self.photon_creation_config[self.photon_creation_map.get('Protonaction_photon')],protonaction_constants)
        
    def neutronaction_step(self):
        """
        Does the neutron exclusive attribute
        """
        #creating a numpy array for constants to give them to neutronaction_cpu that is using Numba
        neutronaction_constants = np.array([self.constants["Prot_Neut_weight1"] , self.constants["Prot_Neut_weight2"] , self.constants["Prot_Neut_threshold"] , self.constants["Prot_Neut_slope"] ])
        (self.particles,self.photons) = neutronaction_cpu(self.particles,self.photons,self.w,self.h,self.photon_creation_config[self.photon_creation_map.get('Neutronaction_photon')],neutronaction_constants)

    def absorption_step(self):
        """
        Does the absorption of photons
        """
        #creating a numpy array for constants to give them to absorption_cpu that is using Numba
        absoprtion_constants = np.array([self.constants["Photon_absorption_probability"]])
        (self.particles,self.photons) = absorption_cpu(self.particles,self.photons,self.w,self.h,absoprtion_constants)

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

@njit(parallel=True)
def collision_cpu(particles :np.ndarray ,photons :np.ndarray ,w,h, create_photon, constants):

    absparticles = np.abs(particles)
    newparticles = np.copy(particles)
    partictot_moving = absparticles[0:6,:,:].sum(axis=0) #(W,H)
    #partictot_rest = absparticles[6,:,:].sum(axis=0)  #(W,H)

    # Particle collision

    
    for x in prange(w):
        for y in prange(h):
            #one-particle interaction (sticking)
            if (partictot_moving[x,y] == 1):
                
                yplus1 = (y+1)%h
                xplus1 = (x+1)%w
                xplus2 = (x+2)%w

                #Defining weights; w1: the particle which incoming particle is facing, w2: two neighbors of w1, w3: two neighbors of w2s, w4: only neighbor of w3s
                w_sum = constants[1] + 2 * constants[2] + 2 * constants[3] + constants[4]

                #normalizing weights so that it is like we always have 6 neighbors included
                w1 = constants[1] * 6 / w_sum
                w2 = constants[2] * 6 / w_sum
                w3 = constants[3] * 6 / w_sum
                w4 = constants[4] * 6 / w_sum
                

                #moving in E direction   
                if (absparticles[0,x,y] == 1):
                    SE = w1 * (absparticles[1,xplus2,y]) + w2 * (absparticles[1,xplus1,y-1] + absparticles[1,xplus1,yplus1]) + w3 * (absparticles[1,x-1,y-1] + absparticles[1,x-1,yplus1]) + w4 * (absparticles[1,x-2,y])
                    SW = w1 * (absparticles[2,xplus2,y]) + w2 * (absparticles[2,xplus1,y-1] + absparticles[2,xplus1,yplus1]) + w3 * (absparticles[2,x-1,y-1] + absparticles[2,x-1,yplus1]) + w4 * (absparticles[2,x-2,y])
                    W = w1 * (absparticles[3,xplus2,y]) + w2 * (absparticles[3,xplus1,y-1] + absparticles[3,xplus1,yplus1]) + w3 * (absparticles[3,x-1,y-1] + absparticles[3,x-1,yplus1]) + w4 * (absparticles[3,x-2,y])
                    NW = w1 * (absparticles[4,xplus2,y]) + w2 * (absparticles[4,xplus1,y-1] + absparticles[4,xplus1,yplus1]) + w3 * (absparticles[4,x-1,y-1] + absparticles[4,x-1,yplus1]) + w4 * (absparticles[4,x-2,y])
                    NE = w1 * (absparticles[5,xplus2,y]) + w2 * (absparticles[5,xplus1,y-1] + absparticles[5,xplus1,yplus1]) + w3 * (absparticles[5,x-1,y-1] + absparticles[5,x-1,yplus1]) + w4 * (absparticles[5,x-2,y])

                    #Finding out which direction is dominant, if any
                    incidence = np.array([SE, SW, W, NW, NE])

                    #Check if only one element is greater than or equal to 'high'
                    condition1 = np.sum(incidence >= constants[5]) == 1

                    #Check if all other elements (excluding the one that meets condition 1) are less than 'low'
                    condition2 = np.all(incidence[incidence < constants[5]] < constants[6])

                    #Find the index of the element that is greater than or equal to 'high'
                    if condition1 and condition2:
                        index_of_bigger = np.where(incidence >= constants[5])[0][0]


                        if np.random.uniform() <= constants[0]:
                            previousdir = 0
                            newdir = index_of_bigger + 1
                            newparticles[previousdir,x,y] = 0
                            newparticles[newdir,x,y] = particles[previousdir,x,y]

                            #creating photon(s) to conserve momentum
                            if (create_photon):
                                momentum_difference = (newdir - previousdir) % 6

                                if(momentum_difference == 0):
                                    photons[previousdir ,x,y] += 1
                                
                                elif(momentum_difference == 1):
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 2):
                                    photons[previousdir ,x,y] += 1
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 3):
                                    photons[previousdir ,x,y] += 2
                                
                                elif(momentum_difference == 4):
                                    photons[previousdir ,x,y] += 1
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 5):
                                    photons[(1 + previousdir) % 6 ,x,y] += 1





                #moving in SE direction   
                if (absparticles[1,x,y] == 1):
                    E = w1 * (absparticles[0,xplus1,yplus1]) + w2 * (absparticles[0,x-1,yplus1] + absparticles[0,xplus2,y]) + w3 * (absparticles[0,x-2,y] +  absparticles[0,xplus1,y-1]) + w4 * (absparticles[0,x-1,y-1])
                    SW = w1 * (absparticles[2,xplus1,yplus1]) + w2 * (absparticles[2,x-1,yplus1] + absparticles[2,xplus2,y]) + w3 * (absparticles[2,x-2,y] +  absparticles[2,xplus1,y-1]) + w4 * (absparticles[2,x-1,y-1])
                    W = w1 * (absparticles[3,xplus1,yplus1]) + w2 * (absparticles[3,x-1,yplus1] + absparticles[3,xplus2,y]) + w3 * (absparticles[3,x-2,y] +  absparticles[3,xplus1,y-1]) + w4 * (absparticles[3,x-1,y-1])
                    NW = w1 *(absparticles[4,xplus1,yplus1]) + w2 * (absparticles[4,x-1,yplus1] + absparticles[4,xplus2,y]) + w3 * (absparticles[4,x-2,y] +  absparticles[4,xplus1,y-1]) + w4 * (absparticles[4,x-1,y-1])
                    NE = w1 * (absparticles[5,xplus1,yplus1]) + w2 * (absparticles[5,x-1,yplus1] + absparticles[5,xplus2,y]) + w3 * (absparticles[5,x-2,y] +  absparticles[5,xplus1,y-1]) + w4 * (absparticles[5,x-1,y-1])

                    #Finding Out which direction is dominant, if any
                    incidence = np.array([E, SW, W, NW, NE])

                    # Check if only one element is greater than or equal to 'high'
                    condition1 = np.sum(incidence >= constants[5]) == 1

                    # Check if all other elements (excluding the one that meets condition 1) are less than 'low'
                    condition2 = np.all(incidence[incidence < constants[5]] < constants[6])

                    # Find the index of the element that is greater than or equal to 'high'
                    if condition1 and condition2:
                        index_of_bigger = np.where(incidence >= constants[5])[0][0]

                        if np.random.uniform() <= constants[0]:
                            previousdir = 1
                            newdir = index_of_bigger
                            if  index_of_bigger > 0:
                                newdir += 1
                            newparticles[previousdir,x,y] = 0
                            newparticles[newdir,x,y] = particles[previousdir,x,y]

                            #creating photon(s) to conserve momentum
                            if (create_photon):
                                momentum_difference = (newdir - previousdir) % 6

                                if(momentum_difference == 0):
                                    photons[previousdir ,x,y] += 1
                                
                                elif(momentum_difference == 1):
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 2):
                                    photons[previousdir ,x,y] += 1
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 3):
                                    photons[previousdir ,x,y] += 2
                                
                                elif(momentum_difference == 4):
                                    photons[previousdir ,x,y] += 1
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 5):
                                    photons[(1 + previousdir) % 6 ,x,y] += 1

                
                # moving in SW direction   
                if (absparticles[2,x,y] == 1):
                    E = w1 * (absparticles[0,x-1,yplus1]) + w2 * (absparticles[0,xplus1,yplus1] + absparticles[0,x-2,y]) + w3 * (absparticles[0,xplus2,y] + absparticles[0,x-1,y-1]) + w4 * (absparticles[0,xplus1,y-1])
                    SE = w1 * (absparticles[1,x-1,yplus1]) + w2 * (absparticles[1,xplus1,yplus1] + absparticles[1,x-2,y]) + w3 * (absparticles[1,xplus2,y] + absparticles[1,x-1,y-1]) + w4 * (absparticles[1,xplus1,y-1])
                    W = w1 * (absparticles[3,x-1,yplus1]) + w2 * (absparticles[3,xplus1,yplus1] + absparticles[3,x-2,y]) + w3 * (absparticles[3,xplus2,y] + absparticles[3,x-1,y-1]) + w4 * (absparticles[3,xplus1,y-1])
                    NW = w1 * (absparticles[4,x-1,yplus1]) + w2 * (absparticles[4,xplus1,yplus1] + absparticles[4,x-2,y]) + w3 * (absparticles[4,xplus2,y] + absparticles[4,x-1,y-1]) + w4 * (absparticles[4,xplus1,y-1])
                    NE = w1 * (absparticles[5,x-1,yplus1]) + w2 * (absparticles[5,xplus1,yplus1] + absparticles[5,x-2,y]) + w3 * (absparticles[5,xplus2,y] + absparticles[5,x-1,y-1]) + w4 * (absparticles[5,xplus1,y-1])

                    #Finding Out which direction is dominant, if any
                    incidence = np.array([E, SE, W, NW, NE])

                    # Check if only one element is greater than or equal to 'high'
                    condition1 = np.sum(incidence >= constants[5]) == 1

                    # Check if all other elements (excluding the one that meets condition 1) are less than 'low'
                    condition2 = np.all(incidence[incidence < constants[5]] < constants[6])

                    # Find the index of the element that is greater than or equal to 'high'
                    if condition1 and condition2:
                        index_of_bigger = np.where(incidence >= constants[5])[0][0]


                        if np.random.uniform() <= constants[0]:
                            previousdir = 2
                            newdir = index_of_bigger
                            if  index_of_bigger > 1:
                                newdir += 1
                            newparticles[previousdir,x,y] = 0
                            newparticles[newdir,x,y] = particles[previousdir,x,y]

                            #creating photon(s) to conserve momentum
                            if (create_photon):
                                momentum_difference = (newdir - previousdir) % 6

                                if(momentum_difference == 0):
                                    photons[previousdir ,x,y] += 1
                                
                                elif(momentum_difference == 1):
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 2):
                                    photons[previousdir ,x,y] += 1
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 3):
                                    photons[previousdir ,x,y] += 2
                                
                                elif(momentum_difference == 4):
                                    photons[previousdir ,x,y] += 1
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 5):
                                    photons[(1 + previousdir) % 6 ,x,y] += 1


                #moving in W direction   
                if (absparticles[3,x,y] == 1):
                    E = w1 * (absparticles[0,x-2,y]) + w2 * (absparticles[0,x-1,yplus1] + absparticles[0,x-1,y-1]) + w3 * (absparticles[0,xplus1,y-1] + absparticles[0,xplus1,yplus1]) + w4 * (absparticles[0,xplus2,y])
                    SE = w1 * (absparticles[1,x-2,y]) + w2 * (absparticles[1,x-1,yplus1] + absparticles[1,x-1,y-1]) + w3 * (absparticles[1,xplus1,y-1] + absparticles[1,xplus1,yplus1]) + w4 * (absparticles[1,xplus2,y])
                    SW = w1 * (absparticles[2,x-2,y]) + w2 * (absparticles[2,x-1,yplus1] + absparticles[2,x-1,y-1]) + w3 * (absparticles[2,xplus1,y-1] + absparticles[2,xplus1,yplus1]) + w4 * (absparticles[2,xplus2,y])
                    NW = w1 * (absparticles[4,x-2,y]) + w2 * (absparticles[4,x-1,yplus1] + absparticles[4,x-1,y-1]) + w3 * (absparticles[4,xplus1,y-1] + absparticles[4,xplus1,yplus1]) + w4 * (absparticles[4,xplus2,y])
                    NE = w1 * (absparticles[5,x-2,y]) + w2 * (absparticles[5,x-1,yplus1] + absparticles[5,x-1,y-1]) + w3 * (absparticles[5,xplus1,y-1] + absparticles[5,xplus1,yplus1]) + w4 * (absparticles[5,xplus2,y])

                    #Finding Out which direction is dominant, if any
                    incidence = np.array([E, SE, SW, NW, NE])

                    # Check if only one element is greater than or equal to 'high'
                    condition1 = np.sum(incidence >= constants[5]) == 1

                    # Check if all other elements (excluding the one that meets condition 1) are less than 'low'
                    condition2 = np.all(incidence[incidence < constants[5]] < constants[6])

                    # Find the index of the element that is greater than or equal to 'high'
                    if condition1 and condition2:
                        index_of_bigger = np.where(incidence >= constants[5])[0][0]


                        if np.random.uniform() <= constants[0]:
                            previousdir = 3
                            newdir = index_of_bigger
                            if  index_of_bigger > 2:
                                newdir += 1
                            newparticles[previousdir,x,y] = 0
                            newparticles[newdir,x,y] = particles[previousdir,x,y]

                            #creating photon(s) to conserve momentum
                            if (create_photon):
                                momentum_difference = (newdir - previousdir) % 6

                                if(momentum_difference == 0):
                                    photons[previousdir ,x,y] += 1
                                
                                elif(momentum_difference == 1):
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 2):
                                    photons[previousdir ,x,y] += 1
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 3):
                                    photons[previousdir ,x,y] += 2
                                
                                elif(momentum_difference == 4):
                                    photons[previousdir ,x,y] += 1
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 5):
                                    photons[(1 + previousdir) % 6 ,x,y] += 1


                #moving in NW direction   
                if (absparticles[4,x,y] == 1):
                    E = w1 * (absparticles[0,x-1,y-1]) + w2 * (absparticles[0,xplus1,y-1] + absparticles[0,x-2,y]) + w3 * (absparticles[0,x-1,yplus1] + absparticles[0,xplus2,y]) + w4 * (absparticles[0,xplus1,yplus1])
                    SE = w1 * (absparticles[1,x-1,y-1]) + w2 * (absparticles[1,xplus1,y-1] + absparticles[1,x-2,y]) + w3 * (absparticles[1,x-1,yplus1] + absparticles[1,xplus2,y]) + w4 * (absparticles[1,xplus1,yplus1])
                    SW = w1 * (absparticles[2,x-1,y-1]) + w2 * (absparticles[2,xplus1,y-1] + absparticles[2,x-2,y]) + w3 * (absparticles[2,x-1,yplus1] + absparticles[2,xplus2,y]) + w4 * (absparticles[2,xplus1,yplus1])
                    W = w1 * (absparticles[3,x-1,y-1]) + w2 * (absparticles[3,xplus1,y-1] + absparticles[3,x-2,y]) + w3 * (absparticles[3,x-1,yplus1] + absparticles[3,xplus2,y]) + w4 * (absparticles[3,xplus1,yplus1])
                    NE = w1 * (absparticles[5,x-1,y-1]) + w2 * (absparticles[5,xplus1,y-1] + absparticles[5,x-2,y]) + w3 * (absparticles[5,x-1,yplus1] + absparticles[5,xplus2,y]) + w4 * (absparticles[5,xplus1,yplus1])

                    #Finding Out which direction is dominant, if any
                    incidence = np.array([E, SE, SW, W, NE])

                    # Check if only one element is greater than or equal to 'high'
                    condition1 = np.sum(incidence >= constants[5]) == 1

                    # Check if all other elements (excluding the one that meets condition 1) are less than 'low'
                    condition2 = np.all(incidence[incidence < constants[5]] < constants[6])

                    # Find the index of the element that is greater than or equal to 'high'
                    if condition1 and condition2:
                        index_of_bigger = np.where(incidence >= constants[5])[0][0]


                        if np.random.uniform() <= constants[0]:
                            previousdir = 4
                            newdir = index_of_bigger
                            if  index_of_bigger > 3:
                                newdir += 1
                            newparticles[previousdir,x,y] = 0
                            newparticles[newdir,x,y] = particles[previousdir,x,y]

                            #creating photon(s) to conserve momentum
                            if (create_photon):
                                momentum_difference = (newdir - previousdir) % 6

                                if(momentum_difference == 0):
                                    photons[previousdir ,x,y] += 1
                                
                                elif(momentum_difference == 1):
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 2):
                                    photons[previousdir ,x,y] += 1
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 3):
                                    photons[previousdir ,x,y] += 2
                                
                                elif(momentum_difference == 4):
                                    photons[previousdir ,x,y] += 1
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 5):
                                    photons[(1 + previousdir) % 6 ,x,y] += 1


                #moving in NE direction   
                if (absparticles[5,x,y] == 1):
                    E = w1 * (absparticles[0,xplus1,y-1]) + w2 * (absparticles[0,x-1,y-1] + absparticles[0,xplus2,y]) + w3 * (absparticles[0,xplus1,yplus1] + absparticles[0,x-2,y]) + w4 * (absparticles[0,x-1,yplus1])
                    SE = w1 * (absparticles[1,xplus1,y-1]) + w2 * (absparticles[1,x-1,y-1] + absparticles[1,xplus2,y]) + w3 * (absparticles[1,xplus1,yplus1] + absparticles[1,x-2,y]) + w4 * (absparticles[1,x-1,yplus1])
                    SW = w1 * (absparticles[2,xplus1,y-1]) + w2 * (absparticles[2,x-1,y-1] + absparticles[2,xplus2,y]) + w3 * (absparticles[2,xplus1,yplus1] + absparticles[2,x-2,y]) + w4 * (absparticles[2,x-1,yplus1])
                    W = w1 * (absparticles[3,xplus1,y-1]) + w2 * (absparticles[3,x-1,y-1] + absparticles[3,xplus2,y]) + w3 * (absparticles[3,xplus1,yplus1] + absparticles[3,x-2,y]) + w4 * (absparticles[3,x-1,yplus1])
                    NW = w1 * (absparticles[4,xplus1,y-1]) + w2 * (absparticles[4,x-1,y-1] + absparticles[4,xplus2,y]) + w3 * (absparticles[4,xplus1,yplus1] + absparticles[4,x-2,y]) + w4 * (absparticles[4,x-1,yplus1])

                    #Finding Out which direction is dominant, if any
                    incidence = np.array([E, SE, SW, W, NW])

                    # Check if only one element is greater than or equal to 'high'
                    condition1 = np.sum(incidence >= constants[5]) == 1

                    # Check if all other elements (excluding the one that meets condition 1) are less than 'low'
                    condition2 = np.all(incidence[incidence < constants[5]] < constants[6])

                    # Find the index of the element that is greater than or equal to 'high'
                    if condition1 and condition2:
                        index_of_bigger = np.where(incidence >= constants[5])[0][0]


                        if np.random.uniform() <= constants[0]:
                            previousdir = 5
                            newdir = index_of_bigger
                            newparticles[previousdir,x,y] = 0
                            newparticles[newdir,x,y] = particles[previousdir,x,y]

                            #creating photon(s) to conserve momentum
                            if (create_photon):
                                momentum_difference = (newdir - previousdir) % 6

                                if(momentum_difference == 0):
                                    photons[previousdir ,x,y] += 1
                                
                                elif(momentum_difference == 1):
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 2):
                                    photons[previousdir ,x,y] += 1
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 3):
                                    photons[previousdir ,x,y] += 2
                                
                                elif(momentum_difference == 4):
                                    photons[previousdir ,x,y] += 1
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
                                
                                elif(momentum_difference == 5):
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
    
    
    return (newparticles,photons)

@njit(parallel = True)
def protonaction_cpu(particles :np.ndarray ,photons :np.ndarray ,w,h, create_photon, constants):

    newparticles = np.copy(particles)
    absparticles = np.abs(particles)
    partictot = absparticles.sum(axis = 0) #(W,H)
    proton = particles == -1
    protontot = proton.sum(axis = 0)
    for x in prange(w):
        for y in prange(h):
            for state in range(7):
                if particles[state,x,y] == -1  and  partictot[x,y] == 1:
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
                        if (create_photon and newdir != previousdir):
                            if(previousdir == 6):
                                photons[(newdir + 3) % 6 ,x,y] += 1
                            
                            else:

                                momentum_difference = (newdir - previousdir) % 6

                                if(momentum_difference == 0):
                                    photons[previousdir ,x,y] += 1
                                    
                                elif(momentum_difference == 1):
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                    
                                elif(momentum_difference == 2):
                                    photons[previousdir ,x,y] += 1
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                    
                                elif(momentum_difference == 3):
                                    photons[previousdir ,x,y] += 2
                                    
                                elif(momentum_difference == 4):
                                    photons[previousdir ,x,y] += 1
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
                                    
                                elif(momentum_difference == 5):
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
            
    return (newparticles,photons)

@njit(parallel = True)
def neutronaction_cpu(particles :np.ndarray ,photons :np.ndarray ,w,h, create_photon, constants):

    newparticles = np.copy(particles)
    absparticles = np.abs(particles)
    partictot = absparticles.sum(axis = 0) #(W,H)
    neutron = particles == 1
    neutrontot = neutron.sum(axis = 0)
    for x in prange(w):
        for y in prange(h):
            for state in range(7):
                if particles[state,x,y] == 1  and  partictot[x,y] == 1:
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
                        if (create_photon and newdir != previousdir):
                            if(previousdir == 6):
                                photons[(newdir + 3) % 6 ,x,y] += 1
                            
                            else:

                                momentum_difference = (newdir - previousdir) % 6

                                if(momentum_difference == 0):
                                    photons[previousdir ,x,y] += 1
                                    
                                elif(momentum_difference == 1):
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                    
                                elif(momentum_difference == 2):
                                    photons[previousdir ,x,y] += 1
                                    photons[(5 + previousdir) % 6 ,x,y] += 1
                                    
                                elif(momentum_difference == 3):
                                    photons[previousdir ,x,y] += 2
                                    
                                elif(momentum_difference == 4):
                                    photons[previousdir ,x,y] += 1
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
                                    
                                elif(momentum_difference == 5):
                                    photons[(1 + previousdir) % 6 ,x,y] += 1
            
    return (newparticles,photons)

@njit(parallel=True)
def propagation_cpu(particles, photons, w,h,dirdico):
    newparticles=np.copy(particles)
    newphotons = np.copy(photons)
    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            for dir in range(6):
                new_position = (loc+dirdico[dir])%np.array([w,h])
                newparticles[dir,new_position[0],new_position[1]] = particles[dir,x,y]
                newphotons[dir,new_position[0],new_position[1]] = photons[dir,x,y]
    return (newparticles,newphotons)

@njit(parallel=True)
def absorption_cpu(particles: np.ndarray, photons: np.ndarray, w,h, constants):

    newparticles = np.copy(particles)
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

                        if(np.random.uniform() < constants[0]):
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
