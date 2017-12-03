"""
Module containing classes and functions for computation of a multimarginal problem for fluids 
"""
import numpy as np


def S(x):
    #return 1.0 - x
    if x < 0.5:
        return 2.0*x 
    else:
        return 2.0 -2.0*x

def generatecost(x,y,sigma):
    """Generates L2 distance cost function

       :param x: vector of coordinates of dimension n 
       :param y: vector of coordinates of dimension n
       :param sigma: regularisation parameter (Sinkhorn)

       :returns cost: cost matrix
    """
    N = len(x)
    cost = np.exp(-(np.tensordot(x,np.ones(N),axes =0) - np.tensordot(np.ones(N),y,axes =0))**2/sigma)
    return cost 

def generatecoupling(x,fun,sigma):
    """Generates L2 distance cost function penalising coupling 

       :param x: vector of coordinates of dimension n 
       :param fun: function defined on scalar x definining the coupling
       :param eps: regularisation parameter (Sinkhorn)

       :returns cost: cost matrix
    """   
    vfun = np.vectorize(fun)
    cost = generatecost(x,vfun(x),sigma)
    return cost   

def setcurrentkernel(k,K):
    """Returns flag which tells if we are at last time step
       
       :param k: current time step (from 0 to K-1)
       :param K: total number of time steps
    """
    return 0 +(k==K-1)

def computepseudomarginal(k,K,G,UMAT):
    """Computes kth pseudo marginal (needs to be multiplied by kth Lagrange multiplier to produce the kth marginal).
       Used for update of kth Lagrange multiplier
  
       :param k: current time step (from 0 to K-1)
       :param K: total number of time steps
       :param G: list G = [Xi0,Xi1]
       :param Xi0: cost associated to successive time steps
       :param Xi1: cost associated to coupling
       :param UMAT: array containing Lagrange multipliers (rows) to enforce marginals 
    """ 
    sequence = np.roll(range(K),-k-1)
    temp_kernel = G[setcurrentkernel(k,K)]
    for ii in sequence[:-1]:
        temp_kernel = (temp_kernel*UMAT[ii,:]).dot(G[setcurrentkernel(ii,K)])

    return np.diag(temp_kernel)

def fixedpoint(LUMAT,G,nu):
   """Fixed point map on Lagrange multipliers for multimarginal problem
       
      :param LUMAT: array containing logarithm of Lagrange multipliers (rows) to enforce marginals
      :param nu: marginal to be enforced at each time
      :param G: list G = [Xi0,Xi1]
      :param Xi0: cost associated to successive time steps
      :param Xi1: cost associated to coupling
      :param nu: marginal to be enforced at each time
   
      :returns LUMAT: updated LUMAT
      :returns err: marginal deviation from previous iteration at time K/2  
   """
   N = LUMAT.shape[1] #Number of cells
   K = LUMAT.shape[0] #Number of time steps
 
   # Change of vaiables
   UMAT = np.exp(LUMAT)
   
   for imod in range(K):
       # Each iteration updates the row (time level) imod in UMAT
       temp = computepseudomarginal(imod,K,G,UMAT)
       if imod == int(K/2):
           err = np.sum(np.abs(temp*UMAT[imod,:]-nu))
        
       UMAT[imod,:] = nu/temp

   LUMAT = np.log(UMAT)
   
   return LUMAT, err


def fixedpoint_old(LUMAT,Xi0,Xi1,nu,h):
   """Fixed point map on Lagrange multipliers for multimarginal problem
       
      :param LUMAT: array containing logarithm of Lagrange multipliers (rows) to enforce marginals
      :param nu: marginal to be enforced at each time
      :param Xi0: cost associated to successive time steps
      :param Xi1: cost associated to coupling

      :returns LUMAT: updated LUMAT
      :returns err: L2 deviation from previous iteration 
   """
   N = LUMAT.shape[1] #Number of cells
   K = LUMAT.shape[0] #Number of time steps
 
   # Change of vaiables
   UMAT = np.exp(LUMAT)
   
   #errv = np.ones(K)

   for imod in range(K):
       # Each iteration updates the row (time level) imod in UMAT

       if imod == K-1:
           Xi_init = Xi1
       else:
           Xi_init = Xi0

       Xiprod = np.eye(N)
       for ll in range(K-1):
            lmod = (ll + imod+1)%K
            if lmod == K-1:
                Xiprod = (Xiprod*UMAT[lmod,:]).dot(Xi1)
            else:
                Xiprod = (Xiprod*UMAT[lmod,:]).dot(Xi0)
       

       update_imod = nu/np.diag(Xi_init.dot(Xiprod))
       #errv[imod] = np.sum(np.abs(UMAT[imod,:]-update_imod))
       if imod == K-1:
           err =  np.sum(np.abs(np.diag(Xi_init.dot(Xiprod))*UMAT[imod,:]-nu))

       UMAT[imod,:] = update_imod
   
   #err = np.sqrt(np.sum(errv**2)/K)
   LUMAT = np.log(UMAT)
   
   return LUMAT, err


def computetransport(UMAT,k_map,G):
    """Compute plan time step 0  -> time step k_map
 
       :param UMAT: array containing Lagrange multipliers (rows) to enforce marginals
       :param k_map: time step at which computing the coupling
       :param G: list G = [Xi0,Xi1]
       :param Xi0: cost associated to successive time steps
       :param Xi1: cost associated to coupling

       :returns T: transport plan   
    """
    N = UMAT.shape[1]
    K = UMAT.shape[0]
    if k_map == 0:
        return eye(N)/N
    else:
        temp_kernel2 = G[0]
        for ss in range(1,k_map):
             temp_kernel2 = (temp_kernel2*UMAT[ss,:]).dot(G[setcurrentkernel(ss,K)])
        temp_kernel = G[setcurrentkernel(k_map,K)]
        for ss in range(k_map+1,K):
             temp_kernel = (temp_kernel*UMAT[ss,:]).dot(G[setcurrentkernel(ss,K)])
        return (temp_kernel*UMAT[0,:])*((temp_kernel2*UMAT[k_map,:]).T)


