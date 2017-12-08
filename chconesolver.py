""" Solves 1D CH equation using cone construction

    :param eps: Sinkhorn regularisation paramter
    :param K: Number of time steps
    :param N: Number of cells in domain
    :param L: Length of domain
"""

import multimarg_fluids as mm
import numpy as np
import matplotlib.pyplot as plt

Nx = 20 
Nr = 5 

L = 1.0
h = L/Nx
hr = np.pi/2.0/(Nr+1)
K = 16 
eps = 0.05

# Parameters for cone metric
a = 1.0
b= 0.5

X0 = np.linspace(0.0,L,Nx)
#X1 = np.asarray([1.0]) 
X1= np.exp(np.linspace(-1.0,1.0,Nr))

X0m,X1m = np.meshgrid(X0,X1)
X =[X0m,X1m]

# Marginal using x1 = arctan(r) change of variable
nu = h*np.ones(Nx) 

Xi0init = mm.generateinitcostcone(X0,X,a,b,eps)
Xi0 = mm.generatecostcone(X,X,a,b,eps)
Xi1 = mm.generatecouplingcone(X,X0,mm.fcone,a,b,eps)

# Alternate coupling assignment via permuation (odd N)
#sigma = np.concatenate((np.arange(0,N,2),np.arange(N-2,0,-2)),axis=0)
#Xi1 = mm.generatecost(X,X[sigma],eps*0.1)

# Lagrange multiplier matrix 
PMAT = np.zeros([K,Nx])

tol = 10**-6
err = 1.0

ii = 0
errv =[]
G = [Xi0init,Xi0,Xi1]
#while err>tol:
for ii in range(100):
    PMAT, err = mm.fixedpointcone(PMAT,G,X1,nu)
    errv.append(err)
    print err


# Compute transport map from 0 to time t
#k_map = int(K/2)
k_map = int(K/2)

Tmap = mm.computetransportcone(PMAT,K-1,X1,[Xi0init,Xi0,Xi1])
plt.imshow(-30*Tmap,origin='lower',cmap = 'gray')

plt.show()





