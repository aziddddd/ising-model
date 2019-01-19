import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib import animation

from numba import jit


# Class to sample an equilibrium state of the Ising model. ( Glauber / Kawasaki )

class Ising():
    def __init__(self, N, temp, method):
        self.N          = N
        self.beta       = 1.0/temp
        self.method     = method
        self.config     = 2*np.random.randint(2, size=(N,N))-1  

    @jit
    def Glauber(self):
        for i in range(self.N):
            for j in range(self.N):
                # Choose a site i at random
                row, col    = np.random.randint(0, self.N), np.random.randint(0, self.N)
                i           = self.config[row, col]

                # Calculate ∆E
                neighbours  = self.config[(row+1)%self.N,col] + self.config[row,(col+1)%self.N] + self.config[(row-1)%self.N,col] + self.config[row,(col-1)%self.N]                     # BOTTOM + RIGHT + LEFT + TOP
                delta_e     = 2*i*neighbours                                                                                                                                            # ∆E = E_nu - E_mu = 2E_nu

                # If ∆E ≤ 0, spin flip always flipped
                if delta_e < 0:
                    i *= -1

                # Else, flipped with P = exp(-∆E/kT)
                elif rand() < np.exp(-delta_e*self.beta):
                    i *= -1
                self.config[row, col] = i
        return self.config
            # Consider the effect of flipping the spin

            # Write down expression of ∆E for the choice site i ( How many spin variables enter this expression)

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

    def simulate(self, i):
        print('Frame : {}'.format(i))
        self.montecarlo()
        X, Y = np.meshgrid(range(self.N), range(self.N))
        quad = plt.pcolormesh(X, Y, self.config, cmap=plt.cm.YlGn)
        plt.title('Time=%d'%i); plt.axis('tight')
        quad.set_array(self.config.ravel())
        return quad
        # plt.close()

    # def animate(self):
    #     plotFigure = plt.figure(figsize=(5, 5), dpi=80)
    #     anim = animation.FuncAnimation(plotFigure, self.simulate, frames=10, interval=50)
    #     return anim

    # def Kawasaki():
        # Choose randomly 2 dinstinct sites i & j

        # Consider the effect of exchanging this pair of spins

        # If E is decreased, the exchange is made with P = 1
        # Else, the exchange is made with P = exp(-∆E/kT)

        # Algorithm for computing E given lattices i and j.

        # Consider the exchange as two consecutive sigle spin flips, so ∆E sum of E changes for 2 moves separately.

# N           = 50
# temp        = .4
# method      = 'glauber'

# ising_model = Ising(N, temp, method)
# plotFigure = plt.figure(figsize=(10, 10), dpi=80)
# ani = animation.FuncAnimation(plotFigure, ising_model.simulate, interval=0)
# plt.show()
