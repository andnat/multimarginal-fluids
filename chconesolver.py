""" Solves 1D CH equation using cone construction

    :param eps: Sinkhorn regularisation paramter
    :param K: Number of time steps
    :param N: Number of cells in domain
    :param L: Length of domain
"""

import multimarg_fluids as mm
import numpy as np
import time as time
import matplotlib.pyplot as plt

Nx = 25 
Nr = 20#7 

L = 1.0
h = L/Nx
hr = np.pi/2.0/(Nr+1)
K = 16 
eps = 0.005

# Parameters for cone metric
a = 1.0
b= 0.5

X0 = np.linspace(0.0,L,Nx)
#X1 = np.asarray([1.0]) 
#X1= np.exp(np.linspace(-2.0,1.0,Nr))
X1 =np.linspace(0.6,1.3,Nr)
#X1 = np.linspace(0.7,1.3,Nr)

X0m,X1m = np.meshgrid(X0,X1)
X =[X0m,X1m]

# Marginal using x1 = arctan(r) change of variable
nu = h*np.ones(Nx) 

Xi0init = mm.generateinitcostcone(X0,X,a,b,eps)
Xi0 = mm.generatecostcone(X,X,a,b,eps)
#Xi1 = mm.generatecouplingcone(X,X0,mm.fcone,a,b,eps)

Xi1 = mm.generatecouplingcone(X,X0,mm.S,a,b,eps,det=False)


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
for ii in range(20000):
    t = time.time()
    PMAT, err = mm.fixedpointcone(PMAT,G,X1,nu)
    errv.append(err)
    elaps= time.time()-t

    print ii
    print elaps
    print err


# Compute transport map from 0 to time t
#k_map = int(K/2)

for k_map in range(K):
    Tmap = mm.computetransportcone(PMAT,k_map,X1,G)
    fig = plt.imshow(-30*Tmap,origin='lower',cmap = 'gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(('transport0_%d.eps' %k_map),format = "eps")

    Tconemap = mm.computetransportcone(PMAT,k_map,X1,G,conedensity_flag=True)
    fig = plt.imshow(-30*Tconemap,origin='lower',cmap = 'gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(('radialmarg0_%d.eps' %k_map),format = "eps")


