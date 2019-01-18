

# Class to sample an equilibrium state of the Ising model. ( Glauber / Kawasaki )

class Glauber():
    # Choose a site i at random

    # Consider the effect of flipping the spin

    # Write down expression of ∆E for the choice site i ( How many spin variables enter this expression)

    # If move performed is ∆E, change the result energy

    # If ∆E ≤ 0, spin flip always flipped
    # Else, flipped with P = exp(-∆E/kT)

class Kawasaki():
    # Choose randomly 2 dinstinct sites i & j

    # Consider the effect of exchanging this pair of spins

    # If E is decreased, the exchange is made with P = 1
    # Else, the exchange is made with P = exp(-∆E/kT)