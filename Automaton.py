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
        self.photons = np.zeros_like(self.particles,dtype=np.int16)
        self.particles = np.where(self.particles>0.9,1,0).astype(np.int16)
        self.particles[:,100:190,40:60]=1


        self.dir = np.array([[0,-1],[-1,0],[0,1],[1,0]])

        self.emission_p = 1.
        self.interaction_p = 0.6

    def collision_step(self):
        self.particles,self.photons= \
            collision_cpu(self.particles,self.photons,self.interaction_p,
                          self.emission_p,self.w,self.h,self.dir)
        

    def propagation_step(self):
        self.particles,self.photons = \
            propagation_cpu(self.particles,self.photons,self.w,self.h,self.dir)
        
        
                    
    def step(self):
        # self.collision_step()
        self.propagation_step()
        self.collision_step()
        self._worldmap[:,:,2]=((self.particles.sum(axis=0)/4.))
        self._worldmap[:,:,:2]=((self.photons.sum(axis=0)/4.))[:,:,None]

@njit(parallel=True)
def collision_cpu(particles,photons,col_prob,emit_prob,w,h,dirdico):
    partictot = particles.sum(axis=0) # (W,H)
    
    # Particle collision
    for x in prange(w):
        for y in prange(h):
            if(partictot[x,y]==2):
                if(particles[0,x,y]==1 and particles[2,x,y]==1):
                    particles[:,x,y]=np.array([0,1,0,1])
                elif(particles[1,x,y]==1 and particles[3,x,y]==1):
                    particles[:,x,y]=np.array([1,0,1,0])
    
    newparticles = np.copy(particles)
    newphotons = np.copy(photons)
    #Particle attraction (photon emission)
    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            dirvec=np.zeros((2,))
            # Weighted direction vector
            for dir in range(4) :
                newpos = (loc+dirdico[dir])%np.array([w,h])
                # Weighted direction vector
                dirvec =dirvec+ dirdico[dir]*partictot[newpos[0],newpos[1]]
            if((dirvec!=0).any()):
                dirnum= get_dir_int(dirvec)#transform to int
                antidirnum = (dirnum+2)%4

                if(particles[antidirnum,x,y]==1 and random.random()<emit_prob):
                    if(particles[dirnum,x,y]==0 and photons[antidirnum,x,y]==0):
                        newparticles[dirnum,x,y]=1
                        newparticles[antidirnum,x,y]=0
                        newphotons[antidirnum,x,y]=1
    particles=np.copy(newparticles)
    photons=np.copy(newphotons)
    #Photon collision
    for x in prange(w):
        for y in prange(h):
            for dir in range(4) :
                antidir=(dir+2)%4
                if(particles[dir,x,y]==1 
                   and particles[antidir,x,y]==0
                   and photons[antidir,x,y]==1):
                    if(random.random()<col_prob):
                        newparticles[dir,x,y]=0
                        newparticles[antidir,x,y]=1
                        newphotons[antidir,x,y]=0

    return newparticles,newphotons

@njit
def get_dir_int(dir_array):
    strength=np.abs(dir_array)

    if(strength[0]>=strength[1]):
       #Its biased for now, if same strength should be random
       if(np.sign(strength[0])>0):
           return 3
       else :
           return 1
    else :
        if(np.sign(strength[1])>0):
            return 2
        else :
            return 0 
    
@njit(parallel=True)
def propagation_cpu(particles,photons,w,h,dirdico):
    newparticles=np.zeros_like(particles)
    newphotons = np.zeros_like(photons)

    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            for dir in range(4):
                newpos = (loc+dirdico[dir])%np.array([w,h])
                newposphot = (loc+2*dirdico[dir])%np.array([w,h])
                if(particles[dir][x,y]==1):
                    newparticles[dir][newpos[0],newpos[1]]=1
                if(photons[dir][x,y]==1):
                    newphotons[dir][newposphot[0],newposphot[1]]=1
    
    return newparticles,newphotons

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