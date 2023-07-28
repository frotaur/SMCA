import numpy as np
from numba import njit, prange,cuda 
import numba.typed as numt
import random
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
        size : 2-uple (W,H)
            Shape of the CA world
        
    """

    def __init__(self,size):
        self.w, self.h  = size
        self.size = size
        # This self._worldmap should be changed in the step function.
        # It should contains floats from 0 to 1 of RGB values.
        self._worldmap = np.random.uniform(size=(self.w,self.h,3))
    

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')
    
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)


class SMCA(Automaton):
    """
        Standard Model Cellular Automaton. Inspired by LGCA.

        Parameters :
        <put them as I go>
    """
    nsteps = 20   # After nsteps the code gives you a statistics of lattice state

    def __init__(self, size, is_countinglumps = True):
        super().__init__(size)
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        self.particles = np.random.randn(4,self.w,self.h) # (4,W,H)
        # self.particles = np.where(self.particles>1.7,1,0).astype(np.int16)
        self.particles = np.where(self.particles>1.7,1,0).astype(np.int16)
        #self.particles[:,100:190,40:60]=1
        self.is_countinglumps = is_countinglumps
        self.steps_cnt = 0
        self.relative_path = "./CSV/"   #The name of folder in which csv files willl be saved  #! You must have a folder with the same name in your project folder
        self.dir = np.array([[0,-1],[-1,0],[0,1],[1,0]])  # Contains arrays of the direction North,West,South,East
        self.rollinput = np.array([[-1,1],[-1,0],[1,1],[1,0]])   # Contains np.roll(,•,•) input for North,West,South,East
        # This part creates two csv files one for average size of the clumps and the other for the histogram:
        if self.is_countinglumps:
            self.filename_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.filename_avglumpsize = self.relative_path + self.filename_timestamp + "_avglumpsize.csv"
            self.filename_lumpsizehist = self.relative_path + self.filename_timestamp + "_lumpsizehistogram.csv"
            with open(self.filename_avglumpsize, 'x') as file:
                pass
            with open(self.filename_lumpsizehist, 'x') as file:
                pass
        
    def propagation_step(self):
        """
            Does the propagation step of the automaton
        """
        # * version 1 of propagation.
        # self.particles = propagation_cpu(self.particles,self.w,self.h,self.dir)
        # * version 2 of propagation. This is much faster but less versatile.
        a = propagation_cpu(self.particles,self.w,self.h,self.dir)
        self.propagation_step_v2()
        if np.subtract(a[0,:,:], self.particles[0,:,:]).any():
            raise ValueError("Propagation methods v1 and v2 yield differen results!")
        
    def propagation_step_v2(self):
        for i in prange(4):
            self.particles[i, :, :] = np.roll(self.particles[i, :, :], self.rollinput[i, 0], self.rollinput[i, 1])


    def collision_step(self):
        """
            Does the collision step of the automaton
        """
        self.particles = collision_cpu(self.particles,self.w,self.h,self.dir)


    def count_lupms(self):
        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            out = list(executor.map(count_lupms_aux, self.particles[[0,1,2,3],:,:]))
        avg_lump_sizes = [tpl[0] for tpl in out]
        bins = [tpl[1] for tpl in out]
        hist = [tpl[2] for tpl in out]
        dirs = [["NORTH"], ["WEST"], ["SOUTH"], ["EAST"]]
        print("="*80)
        print("Step # = " + str(self.steps_cnt))
        print("Averge lump sizes: NORTH, WEST, SOUTH, EAST")
        csv.writer(sys.stdout).writerow(avg_lump_sizes)
        with open(self.filename_avglumpsize, 'a', encoding='UTF8', newline='') as f:
            csv.writer(f).writerow(avg_lump_sizes)
        with open(self.filename_lumpsizehist , 'a', encoding='UTF8', newline='') as f:
            csv.writer(f).writerow([self.steps_cnt])
            for i in prange(4):
                csv.writer(f).writerow(dirs[i])
                csv.writer(f).writerow(bins[i])
                csv.writer(f).writerow(hist[i])
        

    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        self.propagation_step()
        self.collision_step()
        self._worldmap = np.zeros_like(self._worldmap) #(3,W,H)
        self._worldmap[:,:,:]+=((self.particles.sum(axis=0)/4.))[:,:,None]
        if (self.steps_cnt % SMCA.nsteps == 0 and self.is_countinglumps):
            self.count_lupms()
        self.steps_cnt += 1




def count_lupms_aux(colinear_particles):
    # this is required by cv2.connectedComponentsWithStats
    img = np.copy(colinear_particles).astype(np.uint8)
    connectivity = 8 # Neighborhood connectivity (= 4 or 8)
    (num_labels, labeled_img, stats, centroid) = \
        cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    # label=0 is always the background, so we begin from label=1
    avg_size = stats[1:, cv2.CC_STAT_AREA].sum() / (num_labels-1)
    hist = np.bincount(stats[1:, cv2.CC_STAT_AREA])
    bins = np.arange(1, np.max(stats[1:, cv2.CC_STAT_AREA])+1)
    return (avg_size, bins, hist[1:])





@njit(parallel=True)
def collision_cpu(particles :np.ndarray,w,h,dirdico):
    partictot = particles[:].sum(axis=0) # (W,H)
    newparticles = np.copy(particles)
    #natural selection parameter
    n = 10
    #probability of sticking
    p = 1
    # The maximum possible value for the cross section
    simga_max = 1

    # Particle collision
    for x in prange(w):
        for y in prange(h):
            #one-particle sticking interaction
            if (partictot[x,y] == 1):
                yplus1 = (y+1)%h
                xplus1 = (x+1)%w

                #moving in N direction
                if (particles[0,x,y] == 1):
                    S = particles[2,x-1,y-1] + particles[2,x,y-1] + particles[2,xplus1,y-1]
                    W = particles[1,x-1,y-1] + particles[1,x,y-1] + particles[1,xplus1,y-1]
                    E = particles[3,x-1,y-1] + particles[3,x,y-1] + particles[3,xplus1,y-1]
                    
                    if (S >= 2 and W < 2 and E < 2):
                        if np.random.uniform() <= p:
                            newparticles[0,x,y] = 0
                            newparticles[2,x,y] = 1
                    elif (W >= 2 and S < 2 and E < 2):
                        if np.random.uniform() <= p:
                            newparticles[0,x,y] = 0
                            newparticles[1,x,y] = 1
                    elif (E >= 2 and S < 2 and W < 2):
                        if np.random.uniform() <= p:
                            newparticles[0,x,y] = 0
                            newparticles[3,x,y] = 1

                #moving in W direction
                elif (particles[1,x,y] == 1):
                    E = particles[3,x-1,y-1] + particles[3,x-1,y] + particles[3,x-1,yplus1]
                    N = particles[0,x-1,y-1] + particles[0,x-1,y] + particles[0,x-1,yplus1]
                    S = particles[2,x-1,y-1] + particles[2,x-1,y] + particles[2,x-1,yplus1]

                    if (E >= 2 and N < 2 and S < 2):
                        if np.random.uniform() <= p:
                            newparticles[1,x,y] = 0
                            newparticles[3,x,y] = 1
                    elif (N >= 2 and E < 2 and S < 2):
                        if np.random.uniform() <= p:
                            newparticles[1,x,y] = 0
                            newparticles[0,x,y] = 1
                    elif (S >= 2 and E < 2 and N < 2):
                        if np.random.uniform() <= p:
                            newparticles[1,x,y] = 0
                            newparticles[2,x,y] = 1

                #moving in S direction
                elif (particles[2,x,y] == 1):
                    N = particles[0,x-1,yplus1] + particles[0,x,yplus1] + particles[0,xplus1,yplus1]
                    W = particles[1,x-1,yplus1] + particles[1,x,yplus1] + particles[1,xplus1,yplus1]
                    E = particles[3,x-1,yplus1] + particles[3,x,yplus1] + particles[3,xplus1,yplus1]

                    if (N >= 2 and W < 2 and E < 2):
                        if np.random.uniform() <= p:
                            newparticles[2,x,y] = 0
                            newparticles[0,x,y] = 1
                    elif (W >= 2 and N < 2 and E < 2):
                        if np.random.uniform() <= p:
                            newparticles[2,x,y] = 0
                            newparticles[1,x,y] = 1
                    elif (E >= 2 and N < 2 and W < 2):
                        if np.random.uniform() <= p:
                            newparticles[2,x,y] = 0
                            newparticles[3,x,y] = 1

                #moving in E direction
                elif (particles[3,x,y] == 1):
                    W = particles[1,xplus1,y-1] + particles[1,xplus1,y] + particles[1,xplus1,yplus1]
                    N = particles[0,xplus1,y-1] + particles[0,xplus1,y] + particles[0,xplus1,yplus1]
                    S = particles[2,xplus1,y-1] + particles[2,xplus1,y] + particles[2,xplus1,yplus1]

                    if (W >= 2 and N < 2 and S < 2):
                        if np.random.uniform() <= p:
                            newparticles[3,x,y] = 0
                            newparticles[1,x,y] = 1
                    elif (N >= 2 and W < 2 and S < 2):
                        if np.random.uniform() <= p:
                            newparticles[3,x,y] = 0
                            newparticles[0,x,y] = 1
                    elif (S >= 2 and W < 2 and N < 2):
                        if np.random.uniform() <= p:
                            newparticles[3,x,y] = 0
                            newparticles[2,x,y] = 1
            
            #two-particle scattering interaction
            elif(partictot[x,y] == 2):
                #weight of the first neighbour and weight of the second neighbour 
                p1 = 1
                p2 = -0.5
                yplus1 = (y+1)%h
                xplus1 = (x+1)%w
                yplus2 = (y+2)%h
                xplus2 = (x+2)%w
                coherencyN1 = particles[0,x,y-1] + particles[0,x-1,y] + particles[0,x,yplus1] + particles[0,xplus1,y] \
                + particles[0,x-1,y-1] + particles[0,x-1,yplus1] + particles[0,xplus1,y-1] + particles[0,xplus1,yplus1]
                coherencyS1 = particles[2,x,y-1] + particles[2,x-1,y] + particles[2,x,yplus1] + particles[2,xplus1,y] \
                + particles[2,x-1,y-1] + particles[2,x-1,yplus1] + particles[2,xplus1,y-1] + particles[2,xplus1,yplus1]
                coherencyW1 = particles[1,x,y-1] + particles[1,x-1,y] + particles[1,x,yplus1] + particles[1,xplus1,y] \
                + particles[1,x-1,y-1] + particles[1,x-1,yplus1] + particles[1,xplus1,y-1] + particles[1,xplus1,yplus1]
                coherencyE1 = particles[3,x,y-1] + particles[3,x-1,y] + particles[3,x,yplus1] + particles[3,xplus1,y] \
                + particles[3,x-1,y-1] + particles[3,x-1,yplus1] + particles[3,xplus1,y-1] + particles[3,xplus1,yplus1]
                
                coherencyN2 = particles [0,x-2,y-2] + particles[0,x-1,y-2] + particles[0,x,y-2] + particles[0,xplus1,y-2] + particles[0,xplus2,y-2] \
                    + particles[0,x-2,y-1] + particles[0,xplus2,y-1] \
                    + particles[0,x-2,y] + particles[0,xplus2,y] \
                    + particles[0,x-2,yplus1] + particles[0,xplus2,yplus1] \
                    + particles [0,x-2,yplus2] + particles[0,x-1,yplus2] + particles[0,x,yplus2] + particles[0,xplus1,yplus2] + particles[0,xplus2,yplus2]
                coherencyW2 = particles [1,x-2,y-2] + particles[1,x-1,y-2] + particles[1,x,y-2] + particles[1,xplus1,y-2] + particles[1,xplus2,y-2] \
                    + particles[1,x-2,y-1] + particles[1,xplus2,y-1] \
                    + particles[1,x-2,y] + particles[1,xplus2,y] \
                    + particles[1,x-2,yplus1] + particles[1,xplus2,yplus1] \
                    + particles [1,x-2,yplus2] + particles[1,x-1,yplus2] + particles[1,x,yplus2] + particles[1,xplus1,yplus2] + particles[1,xplus2,yplus2]
                coherencyS2 = particles [2,x-2,y-2] + particles[2,x-1,y-2] + particles[2,x,y-2] + particles[2,xplus1,y-2] + particles[2,xplus2,y-2] \
                    + particles[2,x-2,y-1] + particles[2,xplus2,y-1] \
                    + particles[2,x-2,y] + particles[2,xplus2,y] \
                    + particles[2,x-2,yplus1] + particles[2,xplus2,yplus1] \
                    + particles [2,x-2,yplus2] + particles[2,x-1,yplus2] + particles[2,x,yplus2] + particles[2,xplus1,yplus2] + particles[2,xplus2,yplus2]
                coherencyE2 = particles [3,x-2,y-2] + particles[3,x-1,y-2] + particles[3,x,y-2] + particles[3,xplus1,y-2] + particles[3,xplus2,y-2] \
                    + particles[3,x-2,y-1] + particles[3,xplus2,y-1] \
                    + particles[3,x-2,y] + particles[3,xplus2,y] \
                    + particles[3,x-2,yplus1] + particles[3,xplus2,yplus1] \
                    + particles [3,x-2,yplus2] + particles[3,x-1,yplus2] + particles[3,x,yplus2] + particles[3,xplus1,yplus2] + particles[3,xplus2,yplus2]
                totaly = p1*(coherencyN1 - coherencyS1) + p2*(coherencyN2 - coherencyS2)
                totalx = p1*(coherencyE1 - coherencyW1) + p2*(coherencyE2 - coherencyW2)
                s = np.sqrt(totalx**2 + totaly**2)
                #normalized cross section
                sigma = s/(np.sqrt(2)*(np.abs(p1)*8+np.abs(p2)*16))

                if(particles[0,x,y]==1 and particles[2,x,y]==1):
                    #if s == 0 we can not define cos and sin, so we eliminate this situation
                    if s == 0:
                        pass
                    elif (np.random.uniform() <= (np.abs(totalx/s)**n)*sigma):
                        newparticles[1,x,y]=particles[0,x,y]
                        newparticles[3,x,y]=particles[2,x,y]
                        newparticles[0,x,y]=0
                        newparticles[2,x,y]=0
                elif(particles[1,x,y]==1 and particles[3,x,y]==1):
                    #if s == 0 we can not define cos and sin, so we eliminate this situation
                    if s == 0:
                        pass
                    elif(np.random.uniform() <= (np.abs(totaly/s)**n)*sigma):
                        newparticles[0,x,y]=particles[1,x,y]
                        newparticles[2,x,y]=particles[3,x,y]
                        newparticles[1,x,y]=0
                        newparticles[3,x,y]=0
    return newparticles






@njit(parallel=True)
def propagation_cpu(particles,w,h,dirdico):
    newparticles=np.zeros_like(particles)
    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            for dir in range(4):
                newpos = (loc+dirdico[dir])%np.array([w,h])
                if(particles[dir,x,y]==1):
                    newparticles[dir,newpos[0],newpos[1]]=particles[dir,x,y]
    return newparticles








"""
@cuda.jit
def propagation_cuda(partic_t1,partic_t2,dirvecs):
    
        Propagation step in cuda. NOT YET WORKING
        Params : 
        partic_t1 : current state of the world
        partic_t2 : array of zeros, will be filled with the particles
        dirvec : vector of directions

        TODO : share the memory of dirvecs
    
    x,y = cuda.grid(2)
    if(x<partic_t1.shape[0] and y<partic_t1.shape[1]):
        loc = np.array([x,y])
        for dir in range(4):
            if(partic_t1[dir][x][y]==1):
                newpos = (loc+dirvecs[dir])%np.array([w,h])
                partic_t2[dir][newpos[0],newpos[1]]=1

"""