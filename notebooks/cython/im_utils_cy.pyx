from __future__ import division

import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output

from tqdm import tqdm_notebook as tqdm

import os
import warnings


##########################################
cimport cython
cimport numpy as np

from libc.math cimport exp, sqrt, log
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX


warnings.filterwarnings("ignore",category =RuntimeWarning)
warnings.filterwarnings("ignore",category =FutureWarning)


cdef class IsingError(Exception):
    """ An exception class for Ising """
    pass

# Class to sample an equilibrium state of the Ising model. ( Glauber / Kawasaki )
cdef class Ising(object):

    cdef public int N, temp_point, equistep, calcstep
    cdef public str method, RUN_NAME
    cdef public double low_T, high_T, beta, n1, n2, energy, magnet, energy2, magnet2
    cdef public long[:, :] config
    cdef public np.double_t[:] T, E, M, C, X, M_err, C_err

############################################### INITIALISER #####################################################

    def __init__(self, int N, double temp, str method, int temp_point, tuple temp_range, int equistep, int calcstep):
        self.N          = N
        self.method     = method
        self.temp_point = temp_point
        self.beta       = 1.0/temp
        self.config     = 2*np.random.randint(2, size=(N, N))-1
        self.low_T      = temp_range[0]
        self.high_T     = temp_range[1]

        self.RUN_NAME   = 'METHOD_{0} SIZE_{1} TEMP_{2}'.format(method, N, temp)
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

        self.M_err      = np.zeros(temp_point)
        #self.C_array    = np.zeros(calcstep)
        self.C_err      = np.zeros(temp_point)

    def reinitialise(self):
        self.config     = 2*np.random.randint(2, size=(self.N, self.N))-1

        self.T          = np.linspace(self.low_T, self.high_T, self.temp_point)
        self.E          = np.zeros(self.temp_point)
        self.M          = np.zeros(self.temp_point)
        self.C          = np.zeros(self.temp_point)
        self.X          = np.zeros(self.temp_point)

        self.M_err      = np.zeros(self.temp_point)
        #self.C_array    = np.zeros(self.calcstep)
        self.C_err      = np.zeros(self.temp_point)

    def reinitialise_properties(self):
        self.energy     = 0
        self.magnet     = 0
        self.energy2    = 0
        self.magnet2    = 0

############################################## CHOOSE METHOD #####################################################

    def montecarlo(self):
        if self.method == 'glauber':
            self.Glauber()
        elif self.method == 'kawasaki':
            self.Kawasaki()
        else:
            raise IsingError('Unknown method given')
        return self.config

################################################## METHOD ########################################################

    def Glauber(self):
        cdef int i, j, row, col, spin, bottom, right, left, top, neighbour, delta_e
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
                elif rand() < exp(-delta_e*self.beta)*RAND_MAX:
                    spin *= -1
                self.config[row, col] = spin
        return self.config

    def Kawasaki(self):
        cdef int i, j, row_1, row_2, col_1, col_2, spin_1, spin_2, bottom_1, bottom_2, right_1, right_2, left_1, left_2, top_1, top_2, neighbour_1, neighbours_2, delta_e1, delta_e2, delta_e
        for i in range(self.N):
            for j in range(self.N):
                row_1, col_1     = np.random.randint(0, self.N), np.random.randint(0, self.N)
                row_2, col_2     = np.random.randint(0, self.N), np.random.randint(0, self.N)

                spin_1, spin_2   = self.config[row_1, col_1], self.config[row_2, col_2]
                
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
                elif rand() < exp(-delta_e*self.beta)*RAND_MAX:
                    spin_1 *= -1
                    spin_2 *= -1
                self.config[row_1, col_1], self.config[row_2, col_2] = spin_1, spin_2                                          
        return self.config

    def differentLattice(self):
        cdef int spin_1, spin_2, row_1, row_2, col_1, col_2
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
        cdef long[:] display
        self.montecarlo()
        X, Y = np.meshgrid(range(self.N), range(self.N))
        quad = plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn)
        plt.tick_params(axis='both', left=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.title('Time=%d'%i); plt.axis('tight')
        display = self.config.ravel()
        quad.set_array(display)
        return quad

################################################### SNAPSHOTS #######################################################

    def snapshots(self, int numstep):
        cdef int i
        os.makedirs(self.RUN_NAME)
        for i in tqdm(range(numstep)):
            self.montecarlo()
            if i == numstep/numstep :          self.plotStep( i, 1);
            if i == numstep/20      :          self.plotStep( i, 2);
            if i == numstep/10      :          self.plotStep( i, 3);
            if i == numstep/5       :          self.plotStep( i, 4);
            if i == numstep/2       :          self.plotStep( i, 5);
            if i == numstep-1       :          self.plotStep( i, 6);
        plt.savefig('{}/snapshots'.format(self.RUN_NAME))
        self.reinitialise()
                                     
    def plotStep(self, int i, int n_):
        X, Y = np.meshgrid(range(self.N), range(self.N))
        plt.subplot(3, 3, n_, xticklabels=[], yticklabels=[])
        plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn);
        plt.title('Time=%d'%i); plt.axis('tight')
        plt.tight_layout()  
    plt.show()

################################################### DYNAMIC PLOT #####################################################

    def DynamicPlot(self, int numstep):
        cdef int i
        for i in range(numstep):
            self.montecarlo()
            self.dynamicplotStep(i)
        self.reinitialise()
                                     
    def dynamicplotStep(self, int i):
        X, Y = np.meshgrid(range(self.N), range(self.N))
        clear_output(wait=True)
        plt.figure(figsize=(5, 5), dpi=80)   
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn);
        plt.title('Time=%d'%i); plt.axis('tight')
        plt.show()

################################################## STATS ########################################################

    @cython.cdivision(True)
    def analyse(self):
        cdef int tempstep, i, j
        for tempstep in tqdm(range(self.temp_point)):
            self.beta = 1.0/self.T[tempstep]
            
            for i in range(self.equistep):         # equilibrate
                self.montecarlo()                  # Monte Carlo moves

            for j in range(self.calcstep):
                self.montecarlo()
                self.getStats()
            
            self.E[tempstep] = self.n1*self.energy
            self.M[tempstep] = self.n1*self.magnet
            self.C[tempstep] = (self.n1*self.energy2 - self.n2*self.energy*self.energy)*self.beta*self.beta
            self.X[tempstep] = (self.n1*self.magnet2 - self.n2*self.magnet*self.magnet)*self.beta
            # self.C_err[tempstep] = sqrt((self.n1*np.sum(np.asarray(self.C)**2))-(self.n2*np.sum(self.C)**2))
            """
            self.E[tempstep] = self.energy/self.calcstep
            self.M[tempstep] = self.magnet/self.calcstep
            self.C[tempstep] = (self.energy2/self.calcstep - self.energy*self.energy/self.calcstep/self.calcstep)*self.beta*self.beta/self.N
            self.X[tempstep] = (self.magnet2/self.calcstep - self.magnet*self.magnet/self.calcstep/self.calcstep)*self.beta/self.N
            self.C_err[tempstep] = sqrt(np.sum(np.asarray(self.C)**2)-(np.sum(self.C)**2))
            self.M_err[tempstep] = sqrt(2*(self.magnet2-self.magnet)*(-tempstep/log(self.X[tempstep]/self.X[0])/self.calcstep/self.N))
            """
            # self.C_err[tempstep] = sqrt(np.sum((np.asarray(self.C_array) - self.C[tempstep])**2)/self.calcstep)
            self.reinitialise_properties()
        self.dumpData() 
        self.plotStats()
        self.reinitialise()

    def getStats(self):
        self.magnet += self.calcMagnetisation()
        self.energy += self.calcEnergy()
        self.magnet2 += self.calcMagnetisation()**2
        self.energy2 += self.calcEnergy()**2

        """c = (self.n1*self.energy2 - self.n2*self.energy*self.energy)*self.beta*self.beta
        for idx, k in enumerate(self.C_array):
            if j == idx:
                pass
            else:
                self.C_array[idx] += (self.energy2/self.N/self.N/(self.calcstep-1) - self.energy*self.energy/self.N/self.N/(self.calcstep-1)/(self.calcstep-1))*self.beta*self.beta
    def getC_error(self):
        cdef double c_val
        c_val = 0"""

    def calcMagnetisation(self):
        return np.sum(self.config)
     
    def calcEnergy(self):
        cdef int E_config, i, j, spin, bottom, right, left, top, neighbour
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

    def plotStats(self):
        plt.subplot(2, 2, 1 );
        plt.scatter(self.T, self.E, marker='o', s=20, color='RoyalBlue')
        plt.xlabel("Temperature (T)");
        plt.ylabel("Energy ");         plt.axis('tight');

        plt.subplot(2, 2, 2 );
        #plt.errorbar(self.T, np.abs(self.M), yerr=self.X_err, marker='o', color='ForestGreen')
        plt.scatter(self.T, np.abs(self.M), marker='o', s=20, color='ForestGreen')
        plt.xlabel("Temperature (T)"); 
        plt.ylabel("Magnetization ");   plt.axis('tight');

        plt.subplot(2, 2, 3 );
        plt.scatter(self.T, self.C, marker='o', s=20, color='RoyalBlue')
        # plt.errorbar(self.T, self.C, yerr=self.C_err, fmt='.', color='ForestGreen', ecolor='black')
        plt.xlabel("Temperature (T)");  
        plt.ylabel("Specific Heat ");   plt.axis('tight');   

        plt.subplot(2, 2, 4 );
        # plt.errorbar(self.T, self.X, yerr=self.X_err, marker='o', color='ForestGreen')
        plt.scatter(self.T, self.X, marker='o', s=20, color='ForestGreen')
        plt.xlabel("Temperature (T)"); 
        plt.ylabel("Susceptibility");   plt.axis('tight');
        plt.tight_layout()
        plt.savefig('{}/analysis_graph'.format(self.RUN_NAME))

    def dumpData(self):
        np.savetxt('{}/Temperature.txt'.format(self.RUN_NAME), self.T)
        np.savetxt('{}/Energy.txt'.format(self.RUN_NAME), self.E)
        np.savetxt('{}/Magnetization.txt'.format(self.RUN_NAME), self.M)
        # np.savetxt('{}/Specific Heat Error.txt'.format(self.RUN_NAME), self.C_err)
        np.savetxt('{}/Specific Heat.txt'.format(self.RUN_NAME), self.C)
        np.savetxt('{}/Susceptibility.txt'.format(self.RUN_NAME), self.X)