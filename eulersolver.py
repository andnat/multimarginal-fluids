""" Solves 1D Euler equation using Euler1DSolver class

    :param eps: Sinkhorn regularisation paramter
    :param K: Number of time steps
    :param N: Number of cells in domain
    :param L: Length of domain
"""

import multimarg_fluids as mm
import numpy as np
import matplotlib.pyplot as plt

N = 101 
L = 1.0
h = L/N
K = 16 
eps = 0.016**2
nu = np.ones(N)*h 

X = np.linspace(0.0,L,N)
Xi0 = mm.generatecost(X,X,eps)
Xi1 = mm.generatecoupling(X,mm.S,eps*0.1)

# Alternate coupling assignment via permuation (odd N)
#sigma = np.concatenate((np.arange(0,N,2),np.arange(N-2,0,-2)),axis=0)
#Xi1 = mm.generatecost(X,X[sigma],eps*0.1)

# Lagrange multiplier matrix 
LUMAT = np.zeros([K,N])

tol = 10**-6
err = 1.0

ii = 0
errv =[]
G = [Xi0,Xi1]
#while err>tol:
for ii in range(3000):
    LUMAT, err = mm.fixedpoint(LUMAT,G,nu)
    errv.append(err)
    print err


# Compute transport map from 0 to time t
#k_map = int(K/2)
k_map = int(K/2)
#Tmap = mm.computetransport(LUMAT,k_map,Xi0,Xi1)
Tmap = mm.computetransport1(np.exp(LUMAT),k_map,[Xi0,Xi1])
plt.imshow(-30*Tmap,origin='lower',cmap = 'gray')

plt.show()





