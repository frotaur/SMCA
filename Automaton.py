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
    
    nsteps = 20


    def __init__(self, size, is_countinglumps = True):
        super().__init__(size)
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        self.particles = np.random.randn(4,self.w,self.h) # (4,W,H)
        self.particles = np.where(self.particles>1.5,1,0).astype(np.int16)
        #self.particles[:,100:190,40:60]=1
        self.is_countinglumps = is_countinglumps
        self.steps_cnt = 0
        if self.is_countinglumps:
            self.filename_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.filename_avglumpsize = self.filename_timestamp + "_avglumpsize.csv"
            self.filename_lumpsizehist = self.filename_timestamp + "_lumpsizehistogram.csv"
            with open(self.filename_avglumpsize, 'x') as file:
                pass
            with open(self.filename_lumpsizehist, 'x') as file:
                pass
        # Contains in arrays the direction North,West,South,East
        self.dir = np.array([[0,-1],[-1,0],[0,1],[1,0]])
        # Contains np.roll(,•,•) input for North,West,South,East
        self.rollinput = np.array([[-1,0],[-1,1],[1,0],[1,1]])


    def collision_step(self):
        """
            Does the collision step of the automaton
        """
        self.particles= \
            collision_cpu(self.particles,self.w,self.h,self.dir)


    def propagation_step(self):
        """
            Does the propagation step of the automaton
        """
        # self.particles = \
        #     propagation_cpu(self.particles,self.w,self.h,self.dir)
        self.propagation_step_v2()
        
    
    def propagation_step_v2(self):
        for i in prange(4):
            self.particles[i, :, :] = np.roll(self.particles[i, :, :], self.rollinput[i, 0], self.rollinput[i, 1])


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
        if (self.steps_cnt % SMCA.nsteps == 0 & self.is_countinglumps):
            self.count_lupms()
        self.steps_cnt += 1


def count_lupms_aux(colinear_particles):
    # this is required by cv2.connectedComponentsWithStats
    img = np.copy(colinear_particles).astype(np.uint8)
    connectivity = 4 # Neighborhood connectivity (= 4 or 8)
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
    n = 20
    # Particle collision
    for x in prange(w):
        for y in prange(h):
            if(partictot[x,y]==2):
                coherencyN = particles[0,x,y-1] + particles[0,x-1,y] + particles[0,x,y+1] + particles[0,x+1,y]
                coherencyS = particles[2,x,y-1] + particles[2,x-1,y] + particles[2,x,y+1] + particles[2,x+1,y]
                coherencyW = particles[1,x,y-1] + particles[1,x-1,y] + particles[1,x,y+1] + particles[1,x+1,y]
                coherencyE = particles[3,x,y-1] + particles[3,x-1,y] + particles[3,x,y+1] + particles[3,x+1,y]
                totaly = coherencyN - coherencyS
                totalx = coherencyE - coherencyW
                s = np.sqrt(totalx**2 + totaly**2)
                #cross section 
                sigma = s/(4*np.sqrt(2))
                if(particles[0,x,y]==1 and particles[2,x,y]==1):
                    #if s == 0 we can not defive cos and sin, so we eliminate this situation
                    if s == 0:
                        pass
                    elif (np.random.uniform() <= (np.abs(totalx/s)**n)*sigma):
                        newparticles[1,x,y]=particles[0,x,y]
                        newparticles[3,x,y]=particles[2,x,y]
                        newparticles[0,x,y]=0
                        newparticles[2,x,y]=0
                elif(particles[1,x,y]==1 and particles[3,x,y]==1):
                    #if s == 0 we can not defive cos and sin, so we eliminate this situation
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