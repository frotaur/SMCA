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
<<<<<<< Updated upstream
        # Generate random values for self.particles
        self.particles = np.random.randn(4, self.w, self.h)

        # Define thresholds for each direction
        thresholds = np.array([ 1.7, 1.7, 1.7, 1.7])  # Change these values to your desired thresholds

        # Apply thresholds for each layer
        self.particles = np.where(self.particles > thresholds[:, np.newaxis, np.newaxis], 1, 0).astype(np.int16)
=======
        self.particles = np.random.randn(4,self.w,self.h) # (4,W,H)
        self.particles = np.where(self.particles>1.9,1,0).astype(np.int16)

        # self.particles = np.zeros((4,self.w,self.h),dtype=np.int16)
        # self.particles[3,0,self.h//2]=1
        # self.particles[1,self.w-1,self.h//2-1:self.h//2+2]=1
>>>>>>> Stashed changes
        # self.particles[:,100:190,40:60]=1



        # Contains in arrays the direction North,West,South,East
        self.dir = np.array([[0,-1],[-1,0],[0,1],[1,0]])

        # !Invert N and S for propagation only :
        # self.dir = np.array([[0,1], [-1,0], [0,-1], [1,0]])

    def collision_step(self):
        """
            Does the collision step of the automaton
        """
        self.particles= \
            collision_cpu(self.particles,self.w,self.h)
        

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
@njit(parallel=True)
def collision_cpu(particles :np.ndarray,w,h):
    partictot = particles.sum(axis=0) # (W,H)
    newparticles = np.copy(particles)


    # Particle collision
    for x in prange(w):
        for y in prange(h):
            #one-particle sticking interaction
            if (partictot[x,y] == 1):
                # North :
                if(particles[0,x,y]==1):
                    ycheck =  (y-1)%h
                    S =0
                    W =0
                    E =0
                    for xcheck in [(x-1)%w,x%w,(x+1)%w]:
                        S += particles[2,xcheck,ycheck]
                        W += particles[1,xcheck,ycheck]
                        E += particles[3,xcheck,ycheck]
                    
                    count2 = [S>=2, W>=2, E>=2]
                    pos2 = [2, 1, 3]
                    if(sum(count2)==1):# Exactly one is bigger or equaL than two
                        for i in range(3):
                            if(count2[i]):
                                newparticles[0,x,y]=0
                                newparticles[pos2[i],x,y]=1
                # South :
                elif(particles[2,x,y]==1):
                    ycheck =  (y+1)%h
                    N =0
                    W =0
                    E =0
                    for xcheck in [(x-1)%w,x%w,(x+1)%w]:
                        N += particles[0,xcheck,ycheck]
                        W += particles[1,xcheck,ycheck]
                        E += particles[3,xcheck,ycheck]
                    
                    count2 = [N>=2, W>=2, E>=2]
                    pos2 = [0, 1, 3]
                    if(sum(count2)==1):# Exactly one is bigger or equaL than two
                        for i in range(3):
                            if(count2[i]):
                                newparticles[2,x,y]=0
                                newparticles[pos2[i],x,y]=1
                # West :
                elif(particles[1,x,y]==1):
                    xcheck = (x-1)%w
                    N =0
                    S =0
                    E =0
                    for ycheck in [(y-1)%h,y%h,(y+1)%h]:
                        N += particles[0,xcheck,ycheck]
                        S += particles[2,xcheck,ycheck]
                        E += particles[3,xcheck,ycheck]
                    
                    count2 = [N>=2, S>=2, E>=2]
                    pos2 = [0, 2, 3]
                    if(sum(count2)==1):# Exactly one is bigger or equaL than two
                        for i in range(3):
                            if(count2[i]):
                                newparticles[1,x,y]=0
                                newparticles[pos2[i],x,y]=1
                # East :
                elif(particles[3,x,y]==1):
                    # print('checking : ', x,y)
                    # print('Westbois :', particles[1,x])
                    xcheck = (x+1)%w
                    N=0
                    S =0
                    W=0
                    for ycheck in [(y-1)%h,y%h,(y+1)%h]:
                        N+= particles[0,xcheck,ycheck]
                        S += particles[2,xcheck,ycheck]
                        W+= particles[1,xcheck,ycheck]
                    #     print('WESTADD :', particles[1,xcheck,ycheck])
                    # print('WESTBOIS : ', W_4)
        
                    count2 = [N>=2, S>=2, W>=2]
                    pos2 = [0, 2, 1]

                    if(sum(count2)==1):# Exactly one is bigger or equaL than two
                        # print("ALARMS BLAZING")

                        for i in range(3):
                            if(count2[i]):
                                newparticles[3,x,y]=0
                                newparticles[pos2[i],x,y]=1
                else :
                    raise Exception("WHAT DO YOU MEAN ????")
                    
                


    return newparticles

    
@njit(parallel=True)
def propagation_cpu(particles,w,h,dirdico):
    newparticles=np.zeros_like(particles)

    # for x in prange(w):
    #     for y in prange(h):
    #         loc = np.array([x,y])
    #         for dir in range(4):
    #             newpos = (loc+dirdico[dir])%np.array([w,h])
    #             if(particles[dir,x,y]==1):
    #                 newparticles[dir,newpos[0],newpos[1]]=particles[dir,x,y]

    for x in prange(w):
        for y in prange(h):
            for dir in range(4):
                newpos = [(x+dirdico[dir][0])%w,(y+dirdico[dir][1])%h]
                if(particles[dir,x,y]==1):
                    newparticles[dir,newpos[0],newpos[1]]=1
                
    return newparticles
