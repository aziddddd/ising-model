from __future__ import division

import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output

from numba import jit
from tqdm import tqdm_notebook as tqdm

import os
import warnings

warnings.filterwarnings("ignore",category =RuntimeWarning)
warnings.filterwarnings("ignore",category =FutureWarning)

class IsingError(Exception):
    """ An exception class for Ising """
    pass

# Class to sample an equilibrium state of the Ising model. ( Glauber / Kawasaki )
class Ising():
    """
    Class for simulating an Ising Model.
    
    Properties:
    N(int)                      -   system size
    method(str)                 -   dynamics used for simulation ( glauber or kawasaki )
    temp_point(int)             -   number of plot points
    config(long)                -   spin configurations
    beta(float)                 -   1/kBT where kB = 1
    low_T(float)                -   minimum T for plot graph
    high_T(float)               -   maximum T for plot graph

    RUN_NAME(str)               -   data filenames
    equistep(int)               -   equilibrate sweep value
    calcstep(int)               -   measurement sweep value
    factor(int)                 -   actual measurement value

    energy(array)               -   store the sum of energy of spin configuration
    energy2(array)              -   store the sum of energy squared of spin configuration
    magnet(array)               -   store the sum of magnetisation of spin configuration
    magnet2(array)              -   store the sum of magnetisation squared of spin configuration

    T                           -   store plot points for Temperature
    E                           -   store plot points for Energy
    M                           -   store plot points for Magnetisation
    C                           -   store plot points for Heat Capacity
    X                           -   store plot points for Susceptibility

    C_err                       -   store plot points for Uncertainty of Heat Capacity
    
    Methods:
    * reinitialise              -   restore the plotting array to initial state
    * reinitialise_properties   -   restore the calculation array to initial state.
    * montecarlo                -   choose dynamics for simulation
    * Glauber                   -   Glauber dynamics
    * Kawasaki                  -   Kawasaki dynamics
    * differentLattice          -   get different lattices i and j if both positions are the same on first search
    * simulate                  -   animation using FuncAnimation ( Slower as time goes on )
    * snapshots                 -   snapshots of simulation
    * plotStep                  -   plot subplots of snapshots graph
    * dynamicPlot               -   animation using dynamic plotting
    * dynamicplotStep           -   dynamic plotting
    * analyse                   -   analyse the ising model by producing observable stats
    * getStats                  -   obtain the sums of E, E^2, M, M^2
    * calcHeatError             -   calculate the uncertainty ofheat capacity
    * calcMagnetisation         -   calculate the magnetisation given a spin configuration
    * calcEnergy                -   calculate the energy given a spin configuration
    * plotStats                 -   plot the analysis graph
    * dumpData                  -   save the plotting arrays
    """

############################################### INITIALISER #####################################################
    def __init__(self, N, factor, temp, method, temp_point, temp_range, equistep, calcstep):
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
        self.factor     = factor

        # divide by number of samples, and by system size to get intensive values
        self.n1         = 1.0/(calcstep*N*N)
        self.n2         = 1.0/(calcstep*calcstep*N*N)

        self.energy     = np.zeros(factor)
        self.magnet     = np.zeros(factor)
        self.energy2    = np.zeros(factor)
        self.magnet2    = np.zeros(factor)
        self.ci_c       = np.zeros(factor)

        self.T          = np.linspace(temp_range[0], temp_range[1], temp_point)
        self.E          = np.zeros(temp_point)
        self.M          = np.zeros(temp_point)
        self.C          = np.zeros(temp_point)
        self.X          = np.zeros(temp_point)

        self.C_err      = np.zeros(temp_point)

    @jit
    def reinitialise(self):
        self.config     = 2*np.random.randint(2, size=(self.N, self.N))-1

        self.T          = np.linspace(self.low_T, self.high_T, self.temp_point)
        self.E          = np.zeros(self.temp_point)
        self.M          = np.zeros(self.temp_point)
        self.C          = np.zeros(self.temp_point)
        self.X          = np.zeros(self.temp_point)

        self.C_err      = np.zeros(self.temp_point)

    @jit
    def reinitialise_properties(self):
        self.energy     = np.zeros(self.factor)
        self.magnet     = np.zeros(self.factor)
        self.energy2    = np.zeros(self.factor)
        self.magnet2    = np.zeros(self.factor)
        self.ci_c       = np.zeros(self.factor)

############################################## CHOOSE METHOD #####################################################

    @jit
    def montecarlo(self):
        if self.method == 'glauber':
            self.Glauber()
        elif self.method == 'kawasaki':
            self.Kawasaki()
        else:
            raise IsingError('Unknown method given : "{}"'.format(self.method))
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
                row_1, col_1     = np.random.randint(0, self.N), np.random.randint(0, self.N)
                row_2, col_2     = np.random.randint(0, self.N), np.random.randint(0, self.N)

                spin_1, spin_2   = self.config[row_1, col_1], self.config[row_2, col_2]
                
                # In case lattice i and j has the same position, repeat until they are not the same.
                if row_1 == row_2 and col_1 == col_2:
                    ((row_1, col_1), (row_2, col_2)) = self.differentLattice()
                    spin_1, spin_2   = self.config[row_1, col_1], self.config[row_2, col_2]

                # If the spins has the same value, do nothing.
                if spin_1 == spin_2:
                    pass

                else:
                    # Finding nearest neighbour with periodic boundary condition
                    bottom_1, bottom_2   =  self.config[(row_1+1)%self.N, col_1], self.config[(row_2+1)%self.N, col_2]
                    right_1,  right_2    =  self.config[row_1, (col_1+1)%self.N], self.config[row_2, (col_2+1)%self.N]
                    left_1,   left_2     =  self.config[(row_1-1)%self.N, col_1], self.config[(row_2-1)%self.N, col_2]
                    top_1,    top_2      =  self.config[row_1, (col_1-1)%self.N], self.config[row_2, (col_2-1)%self.N]

                    # Considering the effect of swapping between the nearest neighbour spins
                    # If both spins in the same column & next to each other in row
                    if ((abs(row_1 - row_2) == 1) and (col_1 == col_2)):
                        # If spin 1 right to spin 2
                        if row_1 - row_2 == 1:
                            delta_e1 = (spin_1 - top_1)*(bottom_1 + right_1 + left_1)
                            delta_e2 = (spin_2 - bottom_2)*(top_2 + right_2 + left_2)
                        # If spin 1 left to spin 2
                        elif row_1 - row_2 == -1:
                            delta_e1 = (spin_1 - bottom_1)*(top_1 + right_1 + left_1)
                            delta_e2 = (spin_2 - top_2)*(bottom_2 + right_2 + left_2)

                    # If both spins in the same row & next to each other in column
                    elif ((abs(col_1 - col_2) == 1) and (row_1 == row_2)):
                        # If spin 1 below to spin 2
                        if col_1 - col_2 == 1:
                            delta_e1 = (spin_1 - left_1)*(top_1 + bottom_1 + right_1)
                            delta_e2 = (spin_2 - right_2)*(top_2 + bottom_2 + left_2)
                        # If spin 1 above to spin 2
                        elif col_1 - col_2 == -1:
                            delta_e1 = (spin_1 - right_1)*(top_1 + bottom_1 + left_1)
                            delta_e2 = (spin_2 - left_2)*(top_2 + bottom_2 + right_2)
            
                    # If the chosen spins are non-nearest neighbour spins.
                    else:
                        delta_e1 = 2*spin_1*(bottom_1 + right_1 + left_1 + top_1)
                        delta_e2 = 2*spin_2*(bottom_2 + right_2 + left_2 + top_2)

                    delta_e = delta_e1 + delta_e2
                    
                    # If E is decreased, the exchange is made
                    if delta_e < 0:
                        self.config[row_1, col_1], self.config[row_2, col_2] = spin_2, spin_1
                    
                    # Else, the exchange is made with P = exp(-∆E/kT)
                    elif rand() < np.exp(-delta_e*self.beta):
                        self.config[row_1, col_1], self.config[row_2, col_2] = spin_2, spin_1                                                             
        return self.config

    @jit
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
        #os.makedirs(self.RUN_NAME)
        for i in tqdm(range(numstep)):
            self.montecarlo()
            if i == numstep/numstep :          self.plotStep( i, 1);
            if i == numstep/20      :          self.plotStep( i, 2);
            if i == numstep/10      :          self.plotStep( i, 3);
            if i == numstep/5       :          self.plotStep( i, 4);
            if i == numstep/2       :          self.plotStep( i, 5);
            if i == numstep-1       :          self.plotStep( i, 6);
        #plt.savefig('{}/snapshots'.format(self.RUN_NAME))
        self.reinitialise_properties()
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
        self.reinitialise_properties()
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
        for tempstep in tqdm(range(self.temp_point), desc='Temp point', unit='sweepstep'):
            self.beta = 1.0/self.T[tempstep]
            
            for i in tqdm(range(self.equistep), desc='Equilibrate sweep', leave=False, unit='sweep'):         # equilibrate
                self.montecarlo()                                                                             # Monte Carlo moves

            for j in tqdm(range(self.calcstep), desc='Measurement sweep', leave=False, unit='sweep'):         # measurement
                self.montecarlo()                                                                             # Monte Carlo moves
                if (j%10 == 0):
                    self.getStats(int(j/10))                    

            self.E[tempstep] = np.sum(self.energy)/(self.factor)
            self.M[tempstep] = np.sum(self.magnet)/(self.factor)
            self.C[tempstep] = (np.sum(self.energy2)/(self.factor) - np.sum(self.energy)*np.sum(self.energy)/(self.factor)/(self.factor))*self.beta*self.beta/self.N
            self.X[tempstep] = (np.sum(self.magnet2)/(self.factor) - np.sum(self.magnet)*np.sum(self.magnet)/(self.factor)/(self.factor))*self.beta/self.N
            self.C_err[tempstep] = self.calcHeatError(tempstep)

            self.reinitialise_properties()
        #self.dumpData() 
        self.plotStats()
        self.reinitialise()

    @jit
    def getStats(self, j):
        self.magnet[j]  = self.calcMagnetisation()
        self.energy[j]  = self.calcEnergy()
        self.magnet2[j] = self.calcMagnetisation()**2
        self.energy2[j] = self.calcEnergy()**2

    @jit
    def calcHeatError(self, tempstep):
        for idx, i in enumerate(range(self.factor)):
            ci = (np.sum(np.delete(self.energy2, [i]))/(self.factor - 1) - np.sum(np.delete(self.energy, [i]))*np.sum(np.delete(self.energy, [i]))/(self.factor - 1)/(self.factor - 1))*self.beta*self.beta/self.N
            self.ci_c[i] = (ci - self.C[tempstep])**2
        return np.sqrt((self.factor - 1)/(self.factor)*np.sum(self.ci_c))

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
        return E_config

    @jit
    def plotStats(self):
        plt.subplot(2, 2, 1 );
        plt.plot(self.T, self.E, linewidth=0.3, color='black')
        plt.scatter(self.T, self.E, marker='o', s=20, color='RoyalBlue')
        
        plt.xlabel("Temperature (T)");
        plt.ylabel("Energy (E)");         plt.axis('tight');

        plt.subplot(2, 2, 2 );
        plt.plot(self.T, np.abs(self.M), linewidth=0.3, color='black')
        plt.scatter(self.T, np.abs(self.M), marker='o', s=20, color='ForestGreen')
        plt.xlabel("Temperature (T)"); 
        plt.ylabel("Magnetization (M)");   plt.axis('tight');

        plt.subplot(2, 2, 3 );
        plt.plot(self.T, self.C, linewidth=0.3, color='black')
        plt.errorbar(self.T, self.C, yerr=self.C_err, fmt='.', color='ForestGreen', ecolor='black', elinewidth=0.2, capsize=2)
        plt.xlabel("Temperature (T)");  
        plt.ylabel("Specific Heat (C)");   plt.axis('tight');   

        plt.subplot(2, 2, 4 );
        plt.plot(self.T, self.X, linewidth=0.3, color='black')
        plt.scatter(self.T, self.X, marker='o', s=20, color='ForestGreen')
        plt.xlabel("Temperature (T)"); 
        plt.ylabel("Susceptibility (X)");   plt.axis('tight');
        plt.tight_layout()
        #plt.savefig('{}/analysis_graph'.format(self.RUN_NAME))

    @jit
    def dumpData(self):
        np.savetxt('{}/Temperature.txt'.format(self.RUN_NAME), self.T)
        np.savetxt('{}/Energy.txt'.format(self.RUN_NAME), self.E)
        np.savetxt('{}/Magnetization.txt'.format(self.RUN_NAME), self.M)
        np.savetxt('{}/Specific Heat Error.txt'.format(self.RUN_NAME), self.C_err)
        np.savetxt('{}/Specific Heat.txt'.format(self.RUN_NAME), self.C)
        np.savetxt('{}/Susceptibility.txt'.format(self.RUN_NAME), self.X)

