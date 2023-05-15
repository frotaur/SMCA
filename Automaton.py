import numpy as np
from numba import njit, prange
import numba.typed as numt

class Automaton :
    """
        Class that internalizes the rules and evolution of 
        the cellular automaton at hand. It has a step function
        that makes one timestep of the evolution. By convention,
        and to keep in sync with pygame, the world tensor has shape
        (W,H,3). It contains float values between 0 and 1, which
        are mapped to 0 255 when returning output, and describes how the
        world is 'seen' by an observer.

        Parameters :
        size : 2-uple
            Shape of the CA world
        
    """

    def __init__(self,size):
        self.w, self.h  = size
        self.size= size
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
        # 0,1,2,3 are  N,O,S,E directions
        self.particles = np.random.randn(4,self.w,self.h) # (5,W,H)
        self.particles = np.where(self.particles>0.9,1,0).astype(np.int16)
        self.particles[:,100:190,40:60]=1

        self.dir = numt.List([np.array([0,-1]),np.array([-1,0]),np.array([0,1]),np.array([1,0])])

    
    def collision_step(self):
        self.particles=collision_numba(self.particles,self.w,self.h)
        

    def propagation_step(self):
        self.particles=propagation_numba(self.particles,self.w,self.h,self.dir)
        
                    
    def step(self):
        self.collision_step()
        self.propagation_step()

        self._worldmap[:]=((self.particles.sum(axis=0)/4.))[:,:,None]



@njit(parallel=True)
def collision_numba(particles,w,h):
    partictot = particles.sum(axis=0) # (W,H)
    for x in prange(w):
        for y in prange(h):
            if(partictot[x,y]==2):
                if(particles[0][x,y]==1 and particles[2][x,y]==1):
                    particles[:,x,y]=np.array([0,1,0,1])
                elif(particles[1][x,y]==1 and particles[3][x,y]==1):
                    particles[:,x,y]=np.array([1,0,1,0])
    
    return particles

@njit(parallel=True)
def propagation_numba(particles,w,h,dirdico):
    newparticles=np.zeros_like(particles)

    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            for dir in range(4):
                if(particles[dir][x,y]==1):
                    newpos = (loc+dirdico[dir])%np.array([w,h])
                    newparticles[dir][newpos[0],newpos[1]]=1
    
    return newparticles