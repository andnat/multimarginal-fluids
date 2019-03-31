""" Solves 1D CH equation using cone construction

    :param eps: Sinkhorn regularisation paramter
    :param K: Number of time steps
    :param N: Number of cells in domain
    :param L: Length of domain
"""
import sys
import multimarg_fluids as mm
import numpy as np
import time as time
import matplotlib.pyplot as plt

#Save path 
savepath = ""

#Periodic domain
periodic_flag = True

# Discretization parameters 
Nx = 21 # Space discretization 
Nr = 21 # Radial discretization
K = 36 # Time discretization
eps = 0.001 # Entropic regularization
rmin =0.7 # Lower radial bound
rmax= 1.3 # Upper radial bound

# Parameters for cone metric
a = 1.0
b= 0.5
L = 1.0
h = L/Nx

# Discretization
X0 = np.linspace(0.0,L,Nx)
X1 =np.linspace(rmin,rmax,Nr)

#X1 = np.asarray([1.0]) 
X0m,X1m = np.meshgrid(X0,X1)
X =[X0m,X1m]

# Marginal using x1 = arctan(r) change of variable
nu = h*np.ones(Nx) 

# Coputation in logarithmic scale
log_flag = False 

# Generates cost matrices
Xi0init = mm.generateinitcostcone(X0,X,a,b,eps, log_flag=log_flag,periodic_flag=periodic_flag)
Xi0 = mm.generatecostcone(X,X,a,b,eps, log_flag=log_flag,periodic_flag=periodic_flag)

# Generates coupling matrix
Xi1 = mm.generatecouplingcone(X,X0,mm.peakon, mm.peakondet,a,b,eps,periodic_flag=periodic_flag)

# Lagrange multiplier matrix 
PMAT = np.zeros([K,Nx])
errv = []

G = [Xi0init,Xi0,Xi1]

# SINKHORN ITERATIONS
for ii in range(1000):
    t = time.time()
    PMAT, err = mm.fixedpointconeroll(PMAT,G,X1,nu,verbose= False)
    errv.append(err)
    elaps= time.time()-t
    print("ITERATION %d" % ii)
    print("Elapsed time: %f" % elaps)
    print("Marginal error: %f" % err)


# SAVE FIGURES
mm.savefigscone(errv,PMAT, X1, eps ,G, savepath , ext='eps')
mm.savedatacone(errv,PMAT,X1,eps, savepath)

