import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output

from numba import jit
from tqdm import tqdm

class IsingError(Exception):
    """ An exception class for Ising """
    pass

# Class to sample an equilibrium state of the Ising model. ( Glauber / Kawasaki )
class Ising():

############################################### INITIALISER #####################################################

    def __init__(self, N, temp, method):
        self.N          = N
        self.beta       = 1.0/temp
        self.method     = method
        self.config     = 2*np.random.randint(2, size=(N,N))-1  

############################################## CHOOSE METHOD #####################################################

    @jit
    def montecarlo(self):
        if self.method == 'glauber':
            self.Glauber()
        elif self.method == 'kawasaki':
            self.Kawasaki()
        else:
            raise IsingError('Unknown method given')
        return self.config

################################################## METHOD ########################################################

    @jit
    def Glauber(self):
        for i in range(self.N):
            for j in range(self.N):
                # Choose a site i at random
                row, col        = np.random.randint(0, self.N), np.random.randint(0, self.N)
                spin            = self.config[row, col]

                # Finding nearest neighbour with periodic boundary condition                
                bottom  = self.config[(row+1)%self.N, col]
                right   = self.config[row, (col+1)%self.N]
                left    = self.config[(row-1)%self.N, col]
                top     = self.config[row, (col-1)%self.N]

                neighbours      = bottom + right + left + top

                # Calculate ∆E
                delta_e         = 2*spin*neighbours        # ∆E = E_nu - E_mu = 2E_nu                                                                                                               

                # If ∆E ≤ 0, spin flip always flipped
                if delta_e < 0:
                    spin *= -1

                # Else, flipped with P = exp(-∆E/kT)
                elif rand() < np.exp(-delta_e*self.beta):
                    spin *= -1
                self.config[row, col] = spin
        return self.config

    @jit
    def Kawasaki(self):
        for i in range(self.N):
            for j in range(self.N):
                # Choose randomly 2 dinstinct sites i & j
                row_1, col_1     = np.random.randint(0, self.N), np.random.randint(0, self.N)
                row_2, col_2     = np.random.randint(0, self.N), np.random.randint(0, self.N)

                # Only go this function when both spin positions are the same, because jit cant do while loops.
                if row_1 == row_2 and col_1 == col_2:
                    ((row_1, col_1), (row_2, col_2)) = self.differentLattice()

                spin_1, spin_2   = self.config[row_1, col_1], self.config[row_2, col_2]
        
                # Consider the exchange as two consecutive single spin flips
                # Finding nearest neighbour with periodic boundary condition 
                bottom_1, bottom_2   =  self.config[(row_1+1)%self.N, col_1], self.config[(row_2+1)%self.N, col_2]
                right_1,  right_2    =  self.config[row_1, (col_1+1)%self.N], self.config[row_2, (col_2+1)%self.N]
                left_1,   left_2     =  self.config[(row_1-1)%self.N, col_1], self.config[(row_2-1)%self.N, col_2]
                top_1,    top_2      =  self.config[row_1, (col_1-1)%self.N], self.config[row_2, (col_2-1)%self.N]

                neighbours_1, neighbours_2  =  bottom_1 + right_1 + left_1 + top_1, bottom_2 + right_2 + left_2 + top_2

                # Calculate ∆E as a sum of E changes for 2 moves separately.
                delta_e1, delta_e2   = 2*spin_1*neighbours_1, 2*spin_2*neighbours_2
                delta_e              = delta_e1 + delta_e2
                
                # If E is decreased, the exchange is made
                if delta_e < 0:
                    spin_1 *= -1
                    spin_2 *= -1
                
                 # Else, the exchange is made with P = exp(-∆E/kT)
                elif rand() < np.exp(-delta_e*self.beta):
                    spin_1 *= -1
                    spin_2 *= -1
                self.config[row_1, col_1], self.config[row_2, col_2] = spin_1, spin_2                                          
        return self.config

    def differentLattice(self):
        spin_1, spin_2 = 0, 0
        while spin_1 == spin_2:
            # Choose randomly 2 dinstinct sites i & j
            row_1, col_1     = np.random.randint(0, self.N), np.random.randint(0, self.N)
            row_2, col_2     = np.random.randint(0, self.N), np.random.randint(0, self.N)

            spin_1, spin_2   = self.config[row_1, col_1], self.config[row_2, col_2]     
        return ((row_1, col_1), (row_2, col_2))
     
    # @jit
    # def calcEnergy(self, spin, row, col):
    #     bottom  = self.config[(row+1)%self.N, col]
    #     right   = self.config[row, (col+1)%self.N]
    #     left    = self.config[(row-1)%self.N, col]
    #     top     = self.config[row, (col-1)%self.N]

    #     neighbours      = bottom + right + left + top
    #     return spin*neighbours

################################################### ANIMATION #######################################################

    def simulate(self, i):
        print('Frame : {}'.format(i))
        self.montecarlo()
        X, Y = np.meshgrid(range(self.N), range(self.N))
        quad = plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn)
        plt.tick_params(axis='both', left=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
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
        plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
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
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn);
        plt.title('Time=%d'%i); plt.axis('tight')
        plt.show()

# N           = 50                              # System size
# temp        = .4
# method      = 'kawasaki'                      # glauber or kawasaki

# ising_model = Ising(N, temp, method)
# ising_model.snapshots(500)