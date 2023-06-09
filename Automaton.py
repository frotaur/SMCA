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
        self.particles = np.random.randn(4,2,self.w,self.h) # (4,presence+energy,W,H)
        self.photons = np.zeros((4,self.w,self.h),dtype=np.int16)
        self.particles = np.where(self.particles>1.9,1,0).astype(np.int16)
        self.particles[:,100:190,40:60]=1
        self.particles=np.zeros_like(self.particles)
        self.particles[3,:,30,100] = 1
        self.particles[3,1,30,100] = 3
        self.particles[1,:,95,100] = 1
        self.particles[1,1,95,100] = 3


        self.dir = np.array([[0,-1],[-1,0],[0,1],[1,0]])

        self.emission_p = 1.
        self.interaction_p = 1.

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
        
        
        self._worldmap = np.zeros_like(self._worldmap)
        self._worldmap[:,:,1]+=((self.particles[:,0].sum(axis=0)/2.))
        self._worldmap[:,:,:2]+=((self.photons.sum(axis=0)/4.))[:,:,None]

@njit(parallel=True)
def collision_cpu(particles :np.ndarray,photons,col_prob,emit_prob,w,h,dirdico):
    partictot = particles[:,0].sum(axis=0) # (W,H)
    
    # Particle collision
    for x in prange(w):
        for y in prange(h):
            if(partictot[x,y]==2):
                if(particles[0,0,x,y]==1 and particles[2,0,x,y]==1):
                    particles[1,:,x,y]=np.copy(particles[0,:,x,y])
                    particles[3,:,x,y]=np.copy(particles[2,:,x,y])
                elif(particles[1,0,x,y]==1 and particles[3,0,x,y]==1):
                    particles[0,:,x,y]=np.copy(particles[1,:,x,y])
                    particles[2,:,x,y]=np.copy(particles[3,:,x,y])
    
    newparticles = np.copy(particles)
    newphotons = np.copy(photons)
    #Photon collision
    for x in prange(w):
        for y in prange(h):
            for dir in range(4) :
                antidir=(dir+2)%4
                if(particles[dir,0,x,y]==1 
                   and particles[antidir,0,x,y]==0
                   and photons[antidir,x,y]==1):
                    if(random.random()<col_prob):
                        newparticles[dir,:,x,y]=0
                        newparticles[antidir,0,x,y]=1
                        newparticles[antidir,1,x,y]+=1 #Increase the energy by one
                        newphotons[antidir,x,y]=0
    #Particle attraction (photon emission)
    #print('=================COLLISIONS===================')
    particles=np.copy(newparticles)
    photons=np.copy(newphotons)
    partictot = particles[:,0].sum(axis=0) # (W,H)
    particenergy = particles[:,1].sum(axis=0)

    for x in prange(w):
        for y in prange(h):
            loc = np.array([x,y])
            dirvec=np.zeros((2,))
            # Weighted direction vector
            if(partictot[x,y]>0 and particenergy[x,y]>0):
                #print(f'HIT ! There is  particle at : {x,y}')
                for dir in range(4) :
                    newpos = (loc+dirdico[dir])%np.array([w,h])
                    # Weighted direction vector
                    dirvec =dirvec+ dirdico[dir]*partictot[newpos[0],newpos[1]]
                #print(f'DIRVEC : ({dirvec[0],dirvec[1]}')
                if((dirvec!=0).any()):
                    dirnum= get_dir_int(dirvec)#transform to int
                    #print(f'Considering collision for particle : {x,y} with {loc+dirdico[dirnum]}')

                    antidirnum = (dirnum+2)%4
                    if(particles[antidirnum,0,x,y]==1 
                       and random.random()<emit_prob 
                       and particles[antidirnum,1,x,y]>0):
                        if(particles[dirnum,0,x,y]==0 and photons[antidirnum,x,y]==0):
                            #print(f'I EMITTED :{x},{y}')
                            newparticles[dirnum,:,x,y]=np.copy(particles[dirnum,:,x,y])
                            newparticles[dirnum,1,x,y]-=1
                            newparticles[antidirnum,:,x,y]=0
                            newphotons[antidirnum,x,y]=1

    return newparticles,newphotons

@njit
def get_dir_int(dir_array):
    strength=np.abs(dir_array)

    if(strength[0]>=strength[1]):
       #Its biased for now, if same strength should be random
       if(np.sign(dir_array[0])>0):
           # (1,0)
           return 3
       else :
           # (-1,0)
           return 1
    else :
        if(np.sign(dir_array[1])>0):
            # (0,1)
            return 2
        else :
            # (0,-1)
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
                if(particles[dir,0,x,y]==1):
                    if(particles[dir,1,x,y]>0):
                        newparticles[dir,0,newpos[0],newpos[1]]=particles[dir,0,x,y]
                        newparticles[dir,1,newpos[0],newpos[1]]+=particles[dir,1,x,y]
                    else :
                        newparticles[dir,:,loc[0],loc[1]]=np.copy(particles[dir,:,x,y])
                if(photons[dir,x,y]==1):
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