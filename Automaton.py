import numpy as np
from numba import njit, prange,cuda 
import numba.typed as numt
import random


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
        self.size= size
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

    def __init__(self, size):
        super().__init__(size)
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        self.particles = np.random.randn(4,self.w,self.h) # (4,W,H)
        self.particles = np.where(self.particles>1.5,1,0).astype(np.int16)
        self.particles[:,100:190,40:60]=1


        # Contains in arrays the direction North,West,South,East
        self.dir = np.array([[0,-1],[-1,0],[0,1],[1,0]])

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
        self.particles = \
            propagation_cpu(self.particles,self.w,self.h,self.dir)
        
        
                    
    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        self.propagation_step()
        self.collision_step()
        
        
        self._worldmap = np.zeros_like(self._worldmap) #(3,W,H)
        self._worldmap[:,:,:]+=((self.particles.sum(axis=0)/4.))[:,:,None]


@njit(parallel=True)
def collision_cpu(particles :np.ndarray,w,h,dirdico):
    partictot = particles[:].sum(axis=0) # (W,H)
    newparticles = np.copy(particles)
    #prob array dictates the probability of collision. Each components of the prob is correspondent with a specific situation.
    prob = np.array([1, 7/8, 6/8, 5/8, 4/8, 3/8, 2/8, 1/8, 0])
    # Particle collision
    for x in prange(w):
        for y in prange(h):
            if(partictot[x,y]==2):
                if(particles[0,x,y]==1 and particles[2,x,y]==1):
                    coherencyN = particles[0,x,y-1] + particles[0,x-1,y] + particles[0,x,y+1] + particles[0,x+1,y]
                    coherencyS = particles[2,x,y-1] + particles[2,x-1,y] + particles[2,x,y+1] + particles[2,x+1,y]
                    if(np.random.uniform() <= prob[coherencyN + coherencyS]):
                        newparticles[1,x,y]=particles[0,x,y]
                        newparticles[3,x,y]=particles[2,x,y]
                        newparticles[0,x,y]=0
                        newparticles[2,x,y]=0
                elif(particles[1,x,y]==1 and particles[3,x,y]==1):
                    coherencyW = particles[1,x,y-1] + particles[1,x-1,y] + particles[1,x,y+1] + particles[1,x+1,y]
                    coherencyE = particles[3,x,y-1] + particles[3,x-1,y] + particles[3,x,y+1] + particles[3,x+1,y]
                    if(np.random.uniform() <= prob[coherencyW + coherencyE]):
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

@cuda.jit
def propagation_cuda(partic_t1,partic_t2,dirvecs):
    """
        Propagation step in cuda. NOT YET WORKING
        Params : 
        partic_t1 : current state of the world
        partic_t2 : array of zeros, will be filled with the particles
        dirvec : vector of directions

        TODO : share the memory of dirvecs
    """
    x,y = cuda.grid(2)
    if(x<partic_t1.shape[0] and y<partic_t1.shape[1]):
        loc = np.array([x,y])
        for dir in range(4):
            if(partic_t1[dir][x][y]==1):
                newpos = (loc+dirvecs[dir])%np.array([w,h])
                partic_t2[dir][newpos[0],newpos[1]]=1