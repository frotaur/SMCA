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

    def __init__(self, size):
        super().__init__(size)
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        self.particles = np.random.randn(4,self.w,self.h) # (4,W,H)
        self.particles = np.where(self.particles>1.9,1,0).astype(np.int16)
        # self.particles[:,100:190,40:60]=1



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


# ! There is definitely an asymmetry in the code. Someimes alone particles do not stick properly. At the end, usually north and south particles remain.
@njit(parallel=False)
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

                #moving in N direction
                if (particles[0,x,y] == 1):
                    S = particles[2,x-1,y-1] + particles[2,x,y-1] + particles[2,x+1,y-1]
                    W = particles[1,x-1,y-1] + particles[1,x,y-1] + particles[1,x+1,y-1]
                    E = particles[3,x-1,y-1] + particles[3,x,y-1] + particles[3,x+1,y-1]
                    
                    # There is a huge difference between setting S >= 3 and S >= 2
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
                    E = particles[3,x-1,y-1] + particles[3,x-1,y] + particles[3,x-1,y+1]
                    N = particles[0,x-1,y-1] + particles[0,x-1,y] + particles[0,x-1,y+1]
                    S = particles[2,x-1,y-1] + particles[2,x-1,y] + particles[2,x-1,y+1]

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
                    N = particles[0,x-1,y+1] + particles[0,x,y+1] + particles[0,x+1,y+1]
                    W = particles[1,x-1,y+1] + particles[1,x,y+1] + particles[1,x+1,y+1]
                    E = particles[3,x-1,y+1] + particles[3,x,y+1] + particles[3,x+1,y+1]

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
                    W = particles[1,x+1,y-1] + particles[1,x+1,y] + particles[1,x+1,y+1]
                    N = particles[0,x+1,y-1] + particles[0,x+1,y] + particles[0,x+1,y+1]
                    S = particles[2,x+1,y-1] + particles[2,x+1,y] + particles[2,x+1,y+1]

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
                coherencyN = particles[0,x,y-1] + particles[0,x-1,y] + particles[0,x,y+1] + particles[0,x+1,y] \
                + particles[0,x-1,y-1] + particles[0,x-1,y+1] + particles[0,x+1,y-1] + particles[0,x+1,y+1]
                coherencyS = particles[2,x,y-1] + particles[2,x-1,y] + particles[2,x,y+1] + particles[2,x+1,y] \
                + particles[2,x-1,y-1] + particles[2,x-1,y+1] + particles[2,x+1,y-1] + particles[2,x+1,y+1]
                coherencyW = particles[1,x,y-1] + particles[1,x-1,y] + particles[1,x,y+1] + particles[1,x+1,y] \
                + particles[1,x-1,y-1] + particles[1,x-1,y+1] + particles[1,x+1,y-1] + particles[1,x+1,y+1]
                coherencyE = particles[3,x,y-1] + particles[3,x-1,y] + particles[3,x,y+1] + particles[3,x+1,y] \
                + particles[3,x-1,y-1] + particles[3,x-1,y+1] + particles[3,x+1,y-1] + particles[3,x+1,y+1]
                totaly = coherencyN - coherencyS
                totalx = coherencyE - coherencyW
                s = np.sqrt(totalx**2 + totaly**2)
                #non-monotonic cross section 
                #sigma = simga_max*np.abs(np.sin((np.pi/(4*np.sqrt(2)))*s))**3
                #monotonic cross section
                sigma = s/(4*np.sqrt(2))

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

    
@njit(parallel=False)
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