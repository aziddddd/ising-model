import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output

from numba import jit
from tqdm import tqdm

# Class to sample an equilibrium state of the Ising model. ( Glauber / Kawasaki )
class Ising():

############################################### INITIALISER #####################################################

    def __init__(self, N, temp, method):
        self.N          = N
        self.beta       = 1.0/temp
        self.method     = method
        self.config     = 2*np.random.randint(2, size=(N,N))-1  

############################################## CHOOSE METHOD #####################################################

    def montecarlo(self):
        while True:
            if self.method == 'glauber':
                self.Glauber()
                break
            elif self.method == 'kawasaki':
                break
            else:
                self.method = (input('Which method? (glauber/kawasaki)'))
        return self.config

################################################## METHOD ########################################################

    # def Kawasaki():
        # Choose randomly 2 dinstinct sites i & j

        # Consider the effect of exchanging this pair of spins

        # If E is decreased, the exchange is made with P = 1
        # Else, the exchange is made with P = exp(-∆E/kT)

        # Algorithm for computing E given lattices i and j.

        # Consider the exchange as two consecutive sigle spin flips, so ∆E sum of E changes for 2 moves separately.

    @jit
    def Glauber(self):
        for i in range(self.N):
            for j in range(self.N):
                # Choose a site i at random
                row, col    = np.random.randint(0, self.N), np.random.randint(0, self.N)
                i           = self.config[row, col]

                # Calculate ∆E
                neighbours  = self.config[(row+1)%self.N,col] + self.config[row,(col+1)%self.N] + self.config[(row-1)%self.N,col] + self.config[row,(col-1)%self.N]         # BOTTOM + RIGHT + LEFT + TOP
                delta_e     = 2*i*neighbours                                                                                                                                # ∆E = E_nu - E_mu = 2E_nu

                # If ∆E ≤ 0, spin flip always flipped
                if delta_e < 0:
                    i *= -1

                # Else, flipped with P = exp(-∆E/kT)
                elif rand() < np.exp(-delta_e*self.beta):
                    i *= -1
                self.config[row, col] = i
        return self.config

################################################### ANIMATION #######################################################

    def simulate(self, i):
        print('Frame : {}'.format(i))
        self.montecarlo()
        X, Y = np.meshgrid(range(self.N), range(self.N))
        quad = plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn)
        plt.title('Time=%d'%i); plt.axis('tight')
        quad.set_array(self.config.ravel())
        return quad


################################################### SNAPSHOTS #######################################################

    @jit
    def snapshots(self, numstep):
        for i in tqdm(range(numstep)):
            self.montecarlo()
            if i == numstep/numstep :          self.plotStep( i, 1);
            if i == numstep/20      :          self.plotStep( i, 2);
            if i == numstep/10      :          self.plotStep( i, 3);
            if i == numstep/5       :          self.plotStep( i, 4);
            if i == numstep/2       :          self.plotStep( i, 5);
            if i == numstep-1       :          self.plotStep( i, 6);
                 
    @jit                    
    def plotStep(self, i, n_):
        X, Y = np.meshgrid(range(self.N), range(self.N))
        plt.subplot(3, 3, n_, xticklabels=[], yticklabels=[])
        plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn);
        plt.title('Time=%d'%i); plt.axis('tight')
        plt.tight_layout()
    plt.show()

################################################### DYNAMIC PLOT #####################################################

    @jit
    def DynamicPlot(self, numstep):
        for i in range(numstep):
            self.montecarlo()
            self.dynamicplotStep(i)

    @jit                                     
    def dynamicplotStep(self, i):
        X, Y = np.meshgrid(range(self.N), range(self.N))
        clear_output(wait=True)
        plt.figure(figsize=(5, 5), dpi=80)   
        plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn);
        plt.title('Time=%d'%i); plt.axis('tight')
        plt.show()