from __future__ import division

import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output

from numba import jit
from tqdm import tqdm_notebook as tqdm

import warnings

warnings.filterwarnings("ignore",category =RuntimeWarning)

class IsingError(Exception):
    """ An exception class for Ising """
    pass

# Class to sample an equilibrium state of the Ising model. ( Glauber / Kawasaki )
class Ising():

############################################### INITIALISER #####################################################

    def __init__(self, N, temp, method, temp_point, temp_range, equistep, calcstep):
        self.N          = N
        self.method     = method
        self.temp_point = temp_point
        self.beta       = 1.0/temp
        self.config     = 2*np.random.randint(2, size=(N, N))-1
        self.low_T      = temp_range[0]
        self.high_T     = temp_range[1]

        self.equistep   = equistep
        self.calcstep   = calcstep

        # divide by number of samples, and by system size to get intensive values
        self.n1         = 1.0/(calcstep*N*N)
        self.n2         = 1.0/(calcstep*calcstep*N*N)

        self.energy     = 0
        self.magnet     = 0
        self.energy2    = 0
        self.magnet2    = 0

        self.T          = np.linspace(temp_range[0], temp_range[1], temp_point)
        self.E          = np.zeros(temp_point)
        self.M          = np.zeros(temp_point)
        self.C          = np.zeros(temp_point)
        self.X          = np.zeros(temp_point)

    @jit
    def reinitialise(self):
        self.config     = 2*np.random.randint(2, size=(self.N, self.N))-1

        self.T          = np.linspace(self.low_T, self.high_T, self.temp_point)
        self.E          = np.zeros(self.temp_point)
        self.M          = np.zeros(self.temp_point)
        self.C          = np.zeros(self.temp_point)
        self.X          = np.zeros(self.temp_point)

    @jit
    def reinitialise_properties(self):
        self.energy     = 0
        self.magnet     = 0
        self.energy2    = 0
        self.magnet2    = 0

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
        # Get different lattices i and j in this function, because jit cant do while loops.
        spin_1, spin_2 = 0, 0
        while spin_1 == spin_2:
            # Choose randomly 2 dinstinct sites i & j
            row_1, col_1     = np.random.randint(0, self.N), np.random.randint(0, self.N)
            row_2, col_2     = np.random.randint(0, self.N), np.random.randint(0, self.N)

            spin_1, spin_2   = self.config[row_1, col_1], self.config[row_2, col_2]     
        return ((row_1, col_1), (row_2, col_2))

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
        self.reinitialise()
                 
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
        self.reinitialise()

    @jit                                     
    def dynamicplotStep(self, i):
        X, Y = np.meshgrid(range(self.N), range(self.N))
        clear_output(wait=True)
        plt.figure(figsize=(5, 5), dpi=80)   
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn);
        plt.title('Time=%d'%i); plt.axis('tight')
        plt.show()

################################################## STATS ########################################################

    @jit
    def analyse(self):
        for tempstep in tqdm(range(self.temp_point)):
            self.beta = 1.0/self.T[tempstep]
            
            for i in range(self.equistep):         # equilibrate
                self.montecarlo()                  # Monte Carlo moves

            for i in range(self.calcstep):
                self.montecarlo()
                self.getStats()

            self.E[tempstep] = self.n1*self.energy
            self.M[tempstep] = self.n1*self.magnet
            self.C[tempstep] = (self.n1*self.energy2 - self.n2*self.energy*self.energy)*self.beta*self.beta
            self.X[tempstep] = (self.n1*self.magnet2 - self.n2*self.magnet*self.magnet)*self.beta*self.beta
            self.reinitialise_properties()
        self.plotStats()
        self.reinitialise()

    @jit
    def getStats(self):
        self.magnet += self.calcMagnetisation()
        self.energy += self.calcEnergy()
        self.magnet2 += self.calcMagnetisation()**2
        self.energy2 += self.calcEnergy()**2

    @jit
    def calcMagnetisation(self):
        return np.sum(self.config)
     
    @jit
    def calcEnergy(self):
        E_config = 0
        for i in range(self.N):
            for j in range(self.N):
                spin = self.config[i, j]

                bottom  = self.config[(i+1)%self.N, j]
                right   = self.config[i, (j+1)%self.N]
                left    = self.config[(i-1)%self.N, j]
                top     = self.config[i, (j-1)%self.N]

                neighbours  =   bottom + right + left + top
                E_config   +=   -spin*neighbours        
        return E_config/4

    @jit
    def plotStats(self):
        plt.subplot(2, 2, 1 );
        plt.scatter(self.T, self.E, marker='o', s=5, color='RoyalBlue')
        plt.xlabel("Temperature (T)");
        plt.ylabel("Energy ");         plt.axis('tight');

        plt.subplot(2, 2, 2 );
        plt.scatter(self.T, abs(self.M), marker='o', s=5, color='ForestGreen')
        plt.xlabel("Temperature (T)"); 
        plt.ylabel("Magnetization ");   plt.axis('tight');

        plt.subplot(2, 2, 3 );
        plt.scatter(self.T, self.C, marker='o', s=5, color='RoyalBlue')
        plt.xlabel("Temperature (T)");  
        plt.ylabel("Specific Heat ");   plt.axis('tight');   

        plt.subplot(2, 2, 4 );
        plt.scatter(self.T, self.X, marker='o', s=5, color='ForestGreen')
        plt.xlabel("Temperature (T)"); 
        plt.ylabel("Susceptibility");   plt.axis('tight');
        plt.tight_layout() 