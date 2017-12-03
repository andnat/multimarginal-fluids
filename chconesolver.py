""" Solves 1D CH equation using cone construction

    :param eps: Sinkhorn regularisation paramter
    :param K: Number of time steps
    :param N: Number of cells in domain
    :param L: Length of domain
"""

import multimarg_fluids as mm
import numpy as np
import matplotlib.pyplot as plt

N = 5 
Nr = 20 

L = 1.0
h = L/N
hr = np.pi/2.0/(Nr+1)
K = 16 
eps = 0.01

# Parameters for cone metric
a = 1.0
b= 0.5

X0 = np.linspace(0.0,L,N)
X1 = np.linspace(hr,np.pi/2.0,Nr,endpoint = False)

X0m,X1m = np.meshgrid(X0,X1)
X =[X0m,X1m]

# Marginal using x1 = arctan(r) change of variable
nu = (np.tan(X1m.flatten()))**(-3)*(np.cos(X1m.flatten()))**(2)

Xi0 = mm.generatecostcone(X,X,a,b,eps)
Xi1 = mm.generatecouplingcone(X,mm.fcone,a,b,eps)

# Alternate coupling assignment via permuation (odd N)
#sigma = np.concatenate((np.arange(0,N,2),np.arange(N-2,0,-2)),axis=0)
#Xi1 = mm.generatecost(X,X[sigma],eps*0.1)

# Lagrange multiplier matrix 
LUMAT = np.zeros([K,N*Nr])

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

Tmap = mm.computetransport(np.exp(LUMAT),k_map,[Xi0,Xi1])
plt.imshow(-30*Tmap,origin='lower',cmap = 'gray')

plt.show()





