import numpy as np
from numba import njit, prange,cuda 
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

        self.dir = np.array([[0,-1],[-1,0],[0,1],[1,0]])

    
    def collision_step(self):
        self.particles=collision_cpu(self.particles,self.w,self.h)
        

    def propagation_step(self):
        new_partics = np.copy(self.particles)
        propagation_cuda(self.particles,new_partics,self.dir)
        self.particles=new_partics
        
                    
    def step(self):
        # self.collision_step()
        self.propagation_step()

        self._worldmap[:]=((self.particles.sum(axis=0)/4.))[:,:,None]



@njit(parallel=True)
def collision_cpu(particles,w,h):
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
def propagation_cpu(particles,w,h,dirdico):
    newparticles=np.zeros_like(particles)


    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            for dir in range(4):
                if(particles[dir][x,y]==1):
                    newpos = (loc+dirdico[dir])%np.array([w,h])
                    newparticles[dir][newpos[0],newpos[1]]=1
    
    return newparticles

@cuda.jit
def propagation_cuda(partic_t1,partic_t2,dirvecs):
    """
        Propagation step in cuda.
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