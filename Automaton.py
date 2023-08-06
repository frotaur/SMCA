import numpy as np
from numba import njit, prange,cuda 
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


class SMCA(Automaton):
    """
        Standard Model Cellular Automaton. Inspired by LGCA.

        Parameters :
                    First argument: (W,H) tuple for the size of cellur automaton
                    Second argument: Boolean. It is by default True and If you put False it does not give you the statistics. 
    """

    def __init__(self, size, is_countingclumps = True): # size = (W,H)
        super().__init__(size)
        self.steps_cnt = 0
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        
        #create a lattice in which there are some neutrons and protons randomly. 0: nothing, -1: proton, 1: neutron
        self.particles = np.random.randn(5,self.w,self.h) 
        copy_rest_particles = self.particles[4:5,:,:]
        self.particles = np.append(self.particles, copy_rest_particles, axis = 0) # (6,w,h) in which the last two components are for the rest paticles
        threshhold = 2
        tmp1 = self.particles <= threshhold
        tmp2 = self.particles >= -threshhold
        tmp3 = tmp1 * tmp2
        self.particles = np.where(tmp3,0,self.particles)
        self.particles = np.where(self.particles > threshhold,1,self.particles)
        self.particles = np.where(self.particles < -threshhold,-1,self.particles)
        self.particles = self.particles.astype(np.int16)
        
        self.dir = np.array([[0,-1],[-1,0],[0,1],[1,0]])  # Contains arrays of the direction North,West,South,East
        # This part creates two csv files one for average size of the clumps and the other for the histogram:
        self.is_countingclumps = is_countingclumps
        if self.is_countingclumps:
            self.nsteps = 100   # After n steps the code gives you a statistics of lattice state
            self.relative_path = "./CSV/"   #The name of folder in which csv files willl be saved  #! You must have a folder with the same name in your project folder
            self.filename_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.filename_avglumpsize = self.relative_path + self.filename_timestamp + "_avglumpsize.csv"
            self.filename_lumpsizehist = self.relative_path + self.filename_timestamp + "_lumpsizehistogram.csv"
            with open(self.filename_avglumpsize, 'x') as file:
                pass
            with open(self.filename_lumpsizehist, 'x') as file:
                pass
    
    
    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        self.propagation_step()
        self.collision_step()
        self._worldmap = np.zeros_like(self._worldmap) #(W,H,3)
        self.neutron = np.where(self.particles == 1, 1, 0)
        self.proton = np.where(self.particles == -1, 1, 0)
        self._worldmap[:,:,2]=(self.neutron.sum(axis = 0)/1.)[:,:]
        self._worldmap[:,:,0]=(self.proton.sum(axis = 0)/1.)[:,:]
        if (self.is_countingclumps and self.steps_cnt % self.nsteps == 0):
            self.count_clupms()
        self.steps_cnt += 1
        
        
    def propagation_step(self):
        """
            Does the propagation step of the automaton
        """
        self.particles = propagation_cpu(self.particles,self.w,self.h,self.dir)


    def collision_step(self):
        """
            Does the collision step of the automaton
        """
        self.particles = collision_cpu(self.particles,self.w,self.h,self.dir)


    def count_clupms(self):
        """
            Gives you the statistics of the lattice state including a file for the average size clumps and a file for the histogram of clumps size
        """
        with concurrent.futures.ThreadPoolExecutor(6) as executor:
            out = list(executor.map(count_clupms_aux, np.abs(self.particles[[0,1,2,3,4,5],:,:])))
        avg_lump_sizes = [tpl[0] for tpl in out]
        bins = [tpl[1] for tpl in out]
        hist = [tpl[2] for tpl in out]
        dirs = [["NORTH"], ["WEST"], ["SOUTH"], ["EAST"], ["REST1"], ["REST2"]]
        print("="*80)
        print("Step # = " + str(self.steps_cnt))
        print("Averge clump sizes: NORTH, WEST, SOUTH, EAST, REST1, REST2")
        csv.writer(sys.stdout).writerow(avg_lump_sizes)
        with open(self.filename_avglumpsize, 'a', encoding='UTF8', newline='') as f:
            csv.writer(f).writerow(avg_lump_sizes)
        with open(self.filename_lumpsizehist , 'a', encoding='UTF8', newline='') as f:
            csv.writer(f).writerow([self.steps_cnt])
            for i in prange(len(self.particles)):
                csv.writer(f).writerow(dirs[i])
                csv.writer(f).writerow(bins[i])
                csv.writer(f).writerow(hist[i])




def count_clupms_aux(colinear_particles):
    # this is required by cv2.connectedComponentsWithStats
    img = np.copy(colinear_particles).astype(np.uint8)
    connectivity = 8 # Neighborhood connectivity (= 4 or 8)
    (num_labels, labeled_img, stats, centroid) = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    # label=0 is always the background, so we begin from label=1
    try:
        avg_size = stats[1:, cv2.CC_STAT_AREA].sum() / (num_labels-1)
        hist = np.bincount(stats[1:, cv2.CC_STAT_AREA])
        bins = np.arange(1, np.max(stats[1:, cv2.CC_STAT_AREA])+1)
    except ValueError:
        avg_size = 0
        bins = hist = [0]
    return (avg_size, bins, hist[1:])



@njit(parallel=True)
def collision_cpu(particles :np.ndarray,w,h,dirdico):
    absparticles = np.abs(particles)
    newparticles = np.copy(particles)
    partictot_moving = absparticles[0:4,:,:].sum(axis=0) # (W,H)
    partictot_rest = absparticles[4:6,:,:].sum(axis=0)  #(W,H)

    #natural selection parameter
    n = 10
    #probability of sticking
    p = 1
    # As m becomes bigger, rest particles become more stable. 
    m = 2
    # l should be [0,+infinite): As l becomes smaller, The probability of getting rest for two particles coming from opposite direction increases
    l = 0.1
    # Particle collision
    for x in prange(w):
        for y in prange(h):
            #one-particle interaction
            if (partictot_moving[x,y] == 1):
                #weight of the first neighbour and weight of the second neighbour 
                p1 = 1
                p2 = 0
                yplus1 = (y+1)%h
                xplus1 = (x+1)%w
                yplus2 = (y+2)%h
                xplus2 = (x+2)%w
                # R means rest
                coherencyN1 = absparticles[0,x,y-1] + absparticles[0,x-1,y] + absparticles[0,x,yplus1] + absparticles[0,xplus1,y] \
                + absparticles[0,x-1,y-1] + absparticles[0,x-1,yplus1] + absparticles[0,xplus1,y-1] + absparticles[0,xplus1,yplus1]
                coherencyS1 = absparticles[2,x,y-1] + absparticles[2,x-1,y] + absparticles[2,x,yplus1] + absparticles[2,xplus1,y] \
                + absparticles[2,x-1,y-1] + absparticles[2,x-1,yplus1] + absparticles[2,xplus1,y-1] + absparticles[2,xplus1,yplus1]
                coherencyW1 = absparticles[1,x,y-1] + absparticles[1,x-1,y] + absparticles[1,x,yplus1] + absparticles[1,xplus1,y] \
                + absparticles[1,x-1,y-1] + absparticles[1,x-1,yplus1] + absparticles[1,xplus1,y-1] + absparticles[1,xplus1,yplus1]
                coherencyE1 = absparticles[3,x,y-1] + absparticles[3,x-1,y] + absparticles[3,x,yplus1] + absparticles[3,xplus1,y] \
                + absparticles[3,x-1,y-1] + absparticles[3,x-1,yplus1] + absparticles[3,xplus1,y-1] + absparticles[3,xplus1,yplus1]
                coherencyR1 = absparticles[4,x,y-1] + absparticles[4,x-1,y] + absparticles[4,x,yplus1] + absparticles[4,xplus1,y] \
                + absparticles[4,x-1,y-1] + absparticles[4,x-1,yplus1] + absparticles[4,xplus1,y-1] + absparticles[4,xplus1,yplus1] \
                + absparticles[5,x,y-1] + absparticles[5,x-1,y] + absparticles[5,x,yplus1] + absparticles[5,xplus1,y] \
                + absparticles[5,x-1,y-1] + absparticles[5,x-1,yplus1] + absparticles[5,xplus1,y-1] + absparticles[5,xplus1,yplus1]
                
                coherencyN2 = absparticles [0,x-2,y-2] + absparticles[0,x-1,y-2] + absparticles[0,x,y-2] + absparticles[0,xplus1,y-2] + absparticles[0,xplus2,y-2] \
                    + absparticles[0,x-2,y-1] + absparticles[0,xplus2,y-1] \
                    + absparticles[0,x-2,y] + absparticles[0,xplus2,y] \
                    + absparticles[0,x-2,yplus1] + absparticles[0,xplus2,yplus1] \
                    + absparticles [0,x-2,yplus2] + absparticles[0,x-1,yplus2] + absparticles[0,x,yplus2] + absparticles[0,xplus1,yplus2] + absparticles[0,xplus2,yplus2]
                coherencyW2 = absparticles [1,x-2,y-2] + absparticles[1,x-1,y-2] + absparticles[1,x,y-2] + absparticles[1,xplus1,y-2] + absparticles[1,xplus2,y-2] \
                    + absparticles[1,x-2,y-1] + absparticles[1,xplus2,y-1] \
                    + absparticles[1,x-2,y] + absparticles[1,xplus2,y] \
                    + absparticles[1,x-2,yplus1] + absparticles[1,xplus2,yplus1] \
                    + absparticles [1,x-2,yplus2] + absparticles[1,x-1,yplus2] + absparticles[1,x,yplus2] + absparticles[1,xplus1,yplus2] + absparticles[1,xplus2,yplus2]
                coherencyS2 = absparticles [2,x-2,y-2] + absparticles[2,x-1,y-2] + absparticles[2,x,y-2] + absparticles[2,xplus1,y-2] + absparticles[2,xplus2,y-2] \
                    + absparticles[2,x-2,y-1] + absparticles[2,xplus2,y-1] \
                    + absparticles[2,x-2,y] + absparticles[2,xplus2,y] \
                    + absparticles[2,x-2,yplus1] + absparticles[2,xplus2,yplus1] \
                    + absparticles [2,x-2,yplus2] + absparticles[2,x-1,yplus2] + absparticles[2,x,yplus2] + absparticles[2,xplus1,yplus2] + absparticles[2,xplus2,yplus2]
                coherencyE2 = absparticles [3,x-2,y-2] + absparticles[3,x-1,y-2] + absparticles[3,x,y-2] + absparticles[3,xplus1,y-2] + absparticles[3,xplus2,y-2] \
                    + absparticles[3,x-2,y-1] + absparticles[3,xplus2,y-1] \
                    + absparticles[3,x-2,y] + absparticles[3,xplus2,y] \
                    + absparticles[3,x-2,yplus1] + absparticles[3,xplus2,yplus1] \
                    + absparticles [3,x-2,yplus2] + absparticles[3,x-1,yplus2] + absparticles[3,x,yplus2] + absparticles[3,xplus1,yplus2] + absparticles[3,xplus2,yplus2]
                coherencyR2 = absparticles [4,x-2,y-2] + absparticles[4,x-1,y-2] + absparticles[4,x,y-2] + absparticles[4,xplus1,y-2] + absparticles[4,xplus2,y-2] \
                    + absparticles[4,x-2,y-1] + absparticles[4,xplus2,y-1] \
                    + absparticles[4,x-2,y] + absparticles[4,xplus2,y] \
                    + absparticles[4,x-2,yplus1] + absparticles[4,xplus2,yplus1] \
                    + absparticles [4,x-2,yplus2] + absparticles[4,x-1,yplus2] + absparticles[4,x,yplus2] + absparticles[4,xplus1,yplus2] + absparticles[4,xplus2,yplus2] \
                    + absparticles [5,x-2,y-2] + absparticles[5,x-1,y-2] + absparticles[5,x,y-2] + absparticles[5,xplus1,y-2] + absparticles[5,xplus2,y-2] \
                    + absparticles[5,x-2,y-1] + absparticles[5,xplus2,y-1] \
                    + absparticles[5,x-2,y] + absparticles[5,xplus2,y] \
                    + absparticles[5,x-2,yplus1] + absparticles[5,xplus2,yplus1] \
                    + absparticles [5,x-2,yplus2] + absparticles[5,x-1,yplus2] + absparticles[5,x,yplus2] + absparticles[5,xplus1,yplus2] + absparticles[5,xplus2,yplus2]
                    
                totaly = p1*(coherencyN1 - coherencyS1) + p2*(coherencyN2 - coherencyS2)
                totalx = p1*(coherencyE1 - coherencyW1) + p2*(coherencyE2 - coherencyW2)
                totalz = p1*coherencyR1 + p2*coherencyR2
                s = np.sqrt(totalx**2 + totaly**2 + totalz**2)
                #normalized cross section
                sigma = s/(np.sqrt(4)*(np.abs(p1)*8+np.abs(p2)*16))
                
                # One particle comes and force two particles to move: 
                if (partictot_rest[x,y] == 2 and np.random.uniform() < (1-np.abs(totalz/s))**m): 

                    if (absparticles[0,x,y] == 1 or absparticles[2,x,y] == 1):
                        newparticles[4,x,y] = 0
                        newparticles[5,x,y] = 0
                        if np.random.uniform() <= 0.5:
                            newparticles[1,x,y] = particles[4,x,y]
                            newparticles[3,x,y] = particles[5,x,y]
                        else: 
                            newparticles[1,x,y] = particles[5,x,y]
                            newparticles[3,x,y] = particles[4,x,y]
                    else:
                        newparticles[4,x,y] = 0
                        newparticles[5,x,y] = 0
                        if np.random.uniform() <= 0.5:
                            newparticles[0,x,y] = particles[4,x,y]
                            newparticles[2,x,y] = particles[5,x,y]
                        else:
                            newparticles[0,x,y] = particles[5,x,y]
                            newparticles[2,x,y] = particles[4,x,y]
                # moving in N direction
                elif (absparticles[0,x,y] == 1):
                    S = absparticles[2,x-1,y-1] + absparticles[2,x,y-1] + absparticles[2,xplus1,y-1]
                    W = absparticles[1,x-1,y-1] + absparticles[1,x,y-1] + absparticles[1,xplus1,y-1]
                    E = absparticles[3,x-1,y-1] + absparticles[3,x,y-1] + absparticles[3,xplus1,y-1]
                    
                    if (S >= 2 and W < 2 and E < 2):
                        if np.random.uniform() <= p:
                            newparticles[0,x,y] = 0
                            newparticles[2,x,y] = particles[0,x,y]
                    elif (W >= 2 and S < 2 and E < 2):
                        if np.random.uniform() <= p:
                            newparticles[0,x,y] = 0
                            newparticles[1,x,y] = particles[0,x,y]
                    elif (E >= 2 and S < 2 and W < 2):
                        if np.random.uniform() <= p:
                            newparticles[0,x,y] = 0
                            newparticles[3,x,y] = particles[0,x,y]

                #moving in W direction
                elif (absparticles[1,x,y] == 1):
                    E = absparticles[3,x-1,y-1] + absparticles[3,x-1,y] + absparticles[3,x-1,yplus1]
                    N = absparticles[0,x-1,y-1] + absparticles[0,x-1,y] + absparticles[0,x-1,yplus1]
                    S = absparticles[2,x-1,y-1] + absparticles[2,x-1,y] + absparticles[2,x-1,yplus1]

                    if (E >= 2 and N < 2 and S < 2):
                        if np.random.uniform() <= p:
                            newparticles[1,x,y] = 0
                            newparticles[3,x,y] = particles[1,x,y]
                    elif (N >= 2 and E < 2 and S < 2):
                        if np.random.uniform() <= p:
                            newparticles[1,x,y] = 0
                            newparticles[0,x,y] = particles[1,x,y]
                    elif (S >= 2 and E < 2 and N < 2):
                        if np.random.uniform() <= p:
                            newparticles[1,x,y] = 0
                            newparticles[2,x,y] = particles[1,x,y]

                #moving in S direction
                elif (absparticles[2,x,y] == 1):
                    N = absparticles[0,x-1,yplus1] + absparticles[0,x,yplus1] + absparticles[0,xplus1,yplus1]
                    W = absparticles[1,x-1,yplus1] + absparticles[1,x,yplus1] + absparticles[1,xplus1,yplus1]
                    E = absparticles[3,x-1,yplus1] + absparticles[3,x,yplus1] + absparticles[3,xplus1,yplus1]

                    if (N >= 2 and W < 2 and E < 2):
                        if np.random.uniform() <= p:
                            newparticles[2,x,y] = 0
                            newparticles[0,x,y] = particles[2,x,y]
                    elif (W >= 2 and N < 2 and E < 2):
                        if np.random.uniform() <= p:
                            newparticles[2,x,y] = 0
                            newparticles[1,x,y] = particles[2,x,y]
                    elif (E >= 2 and N < 2 and W < 2):
                        if np.random.uniform() <= p:
                            newparticles[2,x,y] = 0
                            newparticles[3,x,y] = particles[2,x,y]

                #moving in E direction
                elif (absparticles[3,x,y] == 1):
                    W = absparticles[1,xplus1,y-1] + absparticles[1,xplus1,y] + absparticles[1,xplus1,yplus1]
                    N = absparticles[0,xplus1,y-1] + absparticles[0,xplus1,y] + absparticles[0,xplus1,yplus1]
                    S = absparticles[2,xplus1,y-1] + absparticles[2,xplus1,y] + absparticles[2,xplus1,yplus1]

                    if (W >= 2 and N < 2 and S < 2):
                        if np.random.uniform() <= p:
                            newparticles[3,x,y] = 0
                            newparticles[1,x,y] = particles[3,x,y]
                    elif (N >= 2 and W < 2 and S < 2):
                        if np.random.uniform() <= p:
                            newparticles[3,x,y] = 0
                            newparticles[0,x,y] = particles[3,x,y]
                    elif (S >= 2 and W < 2 and N < 2):
                        if np.random.uniform() <= p:
                            newparticles[3,x,y] = 0
                            newparticles[2,x,y] = particles[3,x,y]
            
            #two particle interaction
            elif(partictot_moving[x,y] == 2):
                #weight of the first neighbour and weight of the second neighbour 
                p1 = 1
                p2 = 0
                yplus1 = (y+1)%h
                xplus1 = (x+1)%w
                yplus2 = (y+2)%h
                xplus2 = (x+2)%w
                # R means rest 
                coherencyN1 = absparticles[0,x,y-1] + absparticles[0,x-1,y] + absparticles[0,x,yplus1] + absparticles[0,xplus1,y] \
                + absparticles[0,x-1,y-1] + absparticles[0,x-1,yplus1] + absparticles[0,xplus1,y-1] + absparticles[0,xplus1,yplus1]
                coherencyS1 = absparticles[2,x,y-1] + absparticles[2,x-1,y] + absparticles[2,x,yplus1] + absparticles[2,xplus1,y] \
                + absparticles[2,x-1,y-1] + absparticles[2,x-1,yplus1] + absparticles[2,xplus1,y-1] + absparticles[2,xplus1,yplus1]
                coherencyW1 = absparticles[1,x,y-1] + absparticles[1,x-1,y] + absparticles[1,x,yplus1] + absparticles[1,xplus1,y] \
                + absparticles[1,x-1,y-1] + absparticles[1,x-1,yplus1] + absparticles[1,xplus1,y-1] + absparticles[1,xplus1,yplus1]
                coherencyE1 = absparticles[3,x,y-1] + absparticles[3,x-1,y] + absparticles[3,x,yplus1] + absparticles[3,xplus1,y] \
                + absparticles[3,x-1,y-1] + absparticles[3,x-1,yplus1] + absparticles[3,xplus1,y-1] + absparticles[3,xplus1,yplus1]
                coherencyR1 = absparticles[4,x,y-1] + absparticles[4,x-1,y] + absparticles[4,x,yplus1] + absparticles[4,xplus1,y] \
                + absparticles[4,x-1,y-1] + absparticles[4,x-1,yplus1] + absparticles[4,xplus1,y-1] + absparticles[4,xplus1,yplus1] \
                + absparticles[5,x,y-1] + absparticles[5,x-1,y] + absparticles[5,x,yplus1] + absparticles[5,xplus1,y] \
                + absparticles[5,x-1,y-1] + absparticles[5,x-1,yplus1] + absparticles[5,xplus1,y-1] + absparticles[5,xplus1,yplus1]
                
                coherencyN2 = absparticles [0,x-2,y-2] + absparticles[0,x-1,y-2] + absparticles[0,x,y-2] + absparticles[0,xplus1,y-2] + absparticles[0,xplus2,y-2] \
                    + absparticles[0,x-2,y-1] + absparticles[0,xplus2,y-1] \
                    + absparticles[0,x-2,y] + absparticles[0,xplus2,y] \
                    + absparticles[0,x-2,yplus1] + absparticles[0,xplus2,yplus1] \
                    + absparticles [0,x-2,yplus2] + absparticles[0,x-1,yplus2] + absparticles[0,x,yplus2] + absparticles[0,xplus1,yplus2] + absparticles[0,xplus2,yplus2]
                coherencyW2 = absparticles [1,x-2,y-2] + absparticles[1,x-1,y-2] + absparticles[1,x,y-2] + absparticles[1,xplus1,y-2] + absparticles[1,xplus2,y-2] \
                    + absparticles[1,x-2,y-1] + absparticles[1,xplus2,y-1] \
                    + absparticles[1,x-2,y] + absparticles[1,xplus2,y] \
                    + absparticles[1,x-2,yplus1] + absparticles[1,xplus2,yplus1] \
                    + absparticles [1,x-2,yplus2] + absparticles[1,x-1,yplus2] + absparticles[1,x,yplus2] + absparticles[1,xplus1,yplus2] + absparticles[1,xplus2,yplus2]
                coherencyS2 = absparticles [2,x-2,y-2] + absparticles[2,x-1,y-2] + absparticles[2,x,y-2] + absparticles[2,xplus1,y-2] + absparticles[2,xplus2,y-2] \
                    + absparticles[2,x-2,y-1] + absparticles[2,xplus2,y-1] \
                    + absparticles[2,x-2,y] + absparticles[2,xplus2,y] \
                    + absparticles[2,x-2,yplus1] + absparticles[2,xplus2,yplus1] \
                    + absparticles [2,x-2,yplus2] + absparticles[2,x-1,yplus2] + absparticles[2,x,yplus2] + absparticles[2,xplus1,yplus2] + absparticles[2,xplus2,yplus2]
                coherencyE2 = absparticles [3,x-2,y-2] + absparticles[3,x-1,y-2] + absparticles[3,x,y-2] + absparticles[3,xplus1,y-2] + absparticles[3,xplus2,y-2] \
                    + absparticles[3,x-2,y-1] + absparticles[3,xplus2,y-1] \
                    + absparticles[3,x-2,y] + absparticles[3,xplus2,y] \
                    + absparticles[3,x-2,yplus1] + absparticles[3,xplus2,yplus1] \
                    + absparticles [3,x-2,yplus2] + absparticles[3,x-1,yplus2] + absparticles[3,x,yplus2] + absparticles[3,xplus1,yplus2] + absparticles[3,xplus2,yplus2]
                coherencyR2 = absparticles [4,x-2,y-2] + absparticles[4,x-1,y-2] + absparticles[4,x,y-2] + absparticles[4,xplus1,y-2] + absparticles[4,xplus2,y-2] \
                    + absparticles[4,x-2,y-1] + absparticles[4,xplus2,y-1] \
                    + absparticles[4,x-2,y] + absparticles[4,xplus2,y] \
                    + absparticles[4,x-2,yplus1] + absparticles[4,xplus2,yplus1] \
                    + absparticles [4,x-2,yplus2] + absparticles[4,x-1,yplus2] + absparticles[4,x,yplus2] + absparticles[4,xplus1,yplus2] + absparticles[4,xplus2,yplus2] \
                    + absparticles [5,x-2,y-2] + absparticles[5,x-1,y-2] + absparticles[5,x,y-2] + absparticles[5,xplus1,y-2] + absparticles[5,xplus2,y-2] \
                    + absparticles[5,x-2,y-1] + absparticles[5,xplus2,y-1] \
                    + absparticles[5,x-2,y] + absparticles[5,xplus2,y] \
                    + absparticles[5,x-2,yplus1] + absparticles[5,xplus2,yplus1] \
                    + absparticles [5,x-2,yplus2] + absparticles[5,x-1,yplus2] + absparticles[5,x,yplus2] + absparticles[5,xplus1,yplus2] + absparticles[5,xplus2,yplus2]
                    
                totaly = p1*(coherencyN1 - coherencyS1) + p2*(coherencyN2 - coherencyS2)
                totalx = p1*(coherencyE1 - coherencyW1) + p2*(coherencyE2 - coherencyW2)
                totalz = p1*coherencyR1 + p2*coherencyR2
                s = np.sqrt(totalx**2 + totaly**2 + totalz**2)
                #normalized cross section
                sigma = s/(np.sqrt(4)*(np.abs(p1)*8+np.abs(p2)*16))
                #two particles with opposite momentums come and rest after the collision: 
                if ((partictot_rest[x,y] == 0) and ((absparticles[0,x,y] == 1 and absparticles[2,x,y] == 1) or (absparticles[1,x,y] == 1 and absparticles[3,x,y] == 1)) and
                    (np.random.uniform() < np.abs(totalz/s)**l)):
                        newparticles[0,x,y] = newparticles[1,x,y] = newparticles[2,x,y] = newparticles[3,x,y] = 0
                        tmp = particles[0,x,y] + particles[1,x,y] + particles[2,x,y] + particles[3,x,y]
                        if tmp == 2:
                            newparticles[4,x,y] = newparticles[5,x,y] = 1
                        elif tmp ==0:
                            if np.random.uniform() <= 0.5:
                                newparticles[4,x,y] = 1
                                newparticles[5,x,y] = -1
                            else:
                                newparticles[4,x,y] = -1
                                newparticles[5,x,y] = 1
                        elif tmp >= -2:
                            newparticles[4,x,y] = newparticles[5,x,y] = -1
                                
                        
                #two vertical or horizontal particales come and go horizonetally or vertically with a probability
                elif(absparticles[0,x,y]==1 and absparticles[2,x,y]==1):
                    if s == 0:
                        newparticles[0,x,y]=0
                        newparticles[2,x,y]=0
                        if np.random.uniform() <= 0.5:
                            newparticles[1,x,y]=particles[0,x,y]
                            newparticles[3,x,y]=particles[2,x,y]
                        else:
                            newparticles[1,x,y]=particles[2,x,y]
                            newparticles[3,x,y]=particles[0,x,y] 
                        
                    elif (np.random.uniform() <= (np.abs(totalx/s)**n)*sigma):
                        newparticles[0,x,y]=0
                        newparticles[2,x,y]=0
                        if np.random.uniform() <= 0.5:
                            newparticles[1,x,y]=particles[0,x,y]
                            newparticles[3,x,y]=particles[2,x,y]
                        else:
                            newparticles[1,x,y]=particles[2,x,y]
                            newparticles[3,x,y]=particles[0,x,y] 
                        
                elif(absparticles[1,x,y]==1 and absparticles[3,x,y]==1):
                    if s == 0:
                        newparticles[1,x,y]=0
                        newparticles[3,x,y]=0
                        if np.random.uniform() <= 0.5:
                            newparticles[0,x,y]=particles[1,x,y]
                            newparticles[2,x,y]=particles[3,x,y]
                        else:
                            newparticles[0,x,y]=particles[3,x,y]
                            newparticles[2,x,y]=particles[1,x,y] 
                        
                    elif (np.random.uniform() <= (np.abs(totalx/s)**n)*sigma):
                        newparticles[1,x,y]=0
                        newparticles[3,x,y]=0
                        if np.random.uniform() <= 0.5:
                            newparticles[0,x,y]=particles[1,x,y]
                            newparticles[2,x,y]=particles[3,x,y]
                        else:
                            newparticles[0,x,y]=particles[3,x,y]
                            newparticles[2,x,y]=particles[1,x,y]  
                    
                        
                #sticking to the rest particles
                elif(absparticles[0,x,y] == 1 and absparticles[1,x,y] == 1 and partictot_rest[x,y] == 0):
                    if ((absparticles[4,x-1,y] + absparticles[4,x,y-1] + absparticles[4,x-1,y-1] + absparticles[5,x-1,y] + absparticles[5,x,y-1] + absparticles[5,x-1,y-1]) >= 4):
                        if np.random.uniform() <= p:
                            newparticles[0,x,y] = newparticles[1,x,y] = 0
                            if np.random.uniform() <= 0.5:
                                newparticles[4,x,y] = particles[0,x,y]
                                newparticles[5,x,y] = particles[1,x,y]
                            else:
                                newparticles[4,x,y] = particles[1,x,y]
                                newparticles[5,x,y] = particles[0,x,y]
                elif(absparticles[1,x,y] == 1 and absparticles[2,x,y] == 1 and partictot_rest[x,y] == 0):
                    if ((absparticles[4,x-1,y] + absparticles[4,x,y+1] + absparticles[4,x-1,y+1] + absparticles[5,x-1,y] + absparticles[5,x,y+1] + absparticles[5,x-1,y+1]) >= 4):
                        if np.random.uniform() <= p:
                            newparticles[1,x,y] = newparticles[2,x,y] = 0
                            if np.random.uniform() <= 0.5:
                                newparticles[4,x,y] = particles[1,x,y]
                                newparticles[5,x,y] = particles[2,x,y]
                            else:
                                newparticles[4,x,y] = particles[2,x,y]
                                newparticles[5,x,y] = particles[1,x,y]
                elif(absparticles[2,x,y] == 1 and absparticles[3,x,y] == 1 and partictot_rest[x,y] == 0):
                    if ((absparticles[4,x+1,y] + absparticles[4,x,y+1] + absparticles[4,x+1,y+1] + absparticles[5,x+1,y] + absparticles[5,x,y+1] + absparticles[5,x+1,y+1]) >= 4):
                        if np.random.uniform() <= p:
                            newparticles[2,x,y] = newparticles[3,x,y] = 0
                            if np.random.uniform() <= 0.5:
                                newparticles[4,x,y] = particles[2,x,y]
                                newparticles[5,x,y] = particles[3,x,y]
                            else:
                                newparticles[4,x,y] = particles[3,x,y]
                                newparticles[5,x,y] = particles[2,x,y]
                elif(absparticles[3,x,y] == 1 and absparticles[0,x,y] == 1 and partictot_rest[x,y] == 0):
                    if ((absparticles[4,x+1,y] + absparticles[4,x,y-1] + absparticles[4,x+1,y-1] + absparticles[5,x+1,y] + absparticles[5,x,y-1] + absparticles[5,x+1,y-1]) >= 4):
                        if np.random.uniform() <= p:
                            newparticles[3,x,y] = newparticles[0,x,y] = 0
                            if np.random.uniform() <= 0.5:
                                newparticles[4,x,y] = particles[3,x,y]
                                newparticles[5,x,y] = particles[0,x,y]
                            else:
                                newparticles[4,x,y] = particles[0,x,y]
                                newparticles[5,x,y] = particles[3,x,y]          
                            
                                        
    return newparticles






@njit(parallel=True)
def propagation_cpu(particles,w,h,dirdico):
    newparticles=np.copy(particles)
    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            for dir in range(4):
                newpos = (loc+dirdico[dir])%np.array([w,h])
                newparticles[dir,newpos[0],newpos[1]] = particles[dir,x,y]
                
    return newparticles