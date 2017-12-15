"""
Module containing classes and functions for computation of a multimarginal problem for fluids 
"""

import numpy as np
import scipy.optimize as so

def S(x):
    return 1.0 - x
    #if x < 0.5:
    #    return 2.0*x 
    #else:
    #    return 2.0 -2.0*x


def generatecost(x,y,sigma,penalty=1.0):
    """Generates L2 distance cost function c(x,y) only 1D

       :param x: vector of coordinates of dimension n 
       :param y: vector of coordinates of dimension n
       :param sigma: regularisation parameter (Sinkhorn)

       :returns cost: cost matrix
    """
    N = len(x)
    cost = np.exp(-penalty*(np.tensordot(x,np.ones(N),axes =0) - np.tensordot(np.ones(N),y,axes =0))**2/sigma)
    return cost 

def generatecoupling(x,fun,sigma):
    """Generates L2 distance cost function penalising coupling 

       :param x: vector of coordinates of dimension n 
       :param fun: function defined on scalar x definining the coupling
       :param eps: regularisation parameter (Sinkhorn)

       :returns cost: cost matrix
    """   
    vfun = np.vectorize(fun)
    cost = generatecost(x,vfun(x),sigma,10.0)
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
        return np.eye(N)/N
    else:
        temp_kernel2 = G[0]
        for ss in range(1,k_map):
             temp_kernel2 = (temp_kernel2*UMAT[ss,:]).dot(G[setcurrentkernel(ss,K)])
        temp_kernel = G[setcurrentkernel(k_map,K)]
        for ss in range(k_map+1,K):
             temp_kernel = (temp_kernel*UMAT[ss,:]).dot(G[setcurrentkernel(ss,K)])
        return (temp_kernel*UMAT[0,:])*((temp_kernel2*UMAT[k_map,:]).T)

"""
Functions for CH solutions on cone : 
    Implementation using m =sqrt(r) change of variable
    - metric on cone  m g + dm^2/m 
    - distance on cone m1 + m2 - 2*sqrt(m1 m2) cos( d(x1,x2) wedge pi) 
    - map (x,m) = (phi,Jac(phi))
"""


def fcone(x):
    return [(np.exp(x)-1.0)/(np.exp(1)-1),np.exp(x)/(np.exp(1)-1)]

def costcone(x0,x1,y0,y1,a,b):
    """ Computes cost on cone 
         
        :param x0: base space component of first point
        :param x1: radial (m) component of first point
        :param y0: base space component of second point
        :param y1: radial (m) component of second point
        :param a,b: parameters for metric, a=1, b=1/2 corresponds to cone metric in standard coordinates
    """


    if (a/(2.0*b)*abs(x0-y0))< np.pi:
        return 4*b**2*(y1+x1-2*np.sqrt(y1*x1)*np.cos(a/(2.0*b)*(x0-y0)))
    else:
        return 4*b**2*(y1+x1+2*np.sqrt(y1*x1))


def generatecostcone(X,Y,a,b,sigma,penalty=1.0):
    """ Generate cost using cone metric
      
        :param X: X[0] flattened meshgrid base space coordinates
                  X[1] \in log ]0,M] flattened meshgrid radial cordinates 
        :param Y: Y[0] flattened meshgrid base space coordinates
                  Y[1] \in log ]0,M] flattened meshgrid radial coordinates
        :param a,b: cone metric parameters
        :param sigma: Sinkhor regularization parameter 
        :param penalty: diminishes sigma, new sigma = sigma/penalty
        :returns cost: cost matrix
    """
    
    Nx = X[0].size
    Ny = Y[0].size
    vcostcone = np.vectorize(costcone)

    cost = np.exp(-penalty*vcostcone(np.tensordot(X[0].flatten(),np.ones(Ny),axes =0),
                            np.tensordot(X[1].flatten(),np.ones(Ny),axes =0),
                            np.tensordot(np.ones(Nx),Y[0].flatten(),axes =0),
                            np.tensordot(np.ones(Nx),Y[1].flatten(),axes =0),a,b)/sigma)
   
    return cost



def generateinitcostcone(x,Y,a,b,sigma):
    """ Generate cost relative to initial time (when radial coordinate is fixed to 1) using cone metric
      
        :param x: array of base space coordinates
        :param Y: Y[0] flattened meshgrid base space coordinates
                  Y[1] \in log ]0,M] flattened meshgrid radial coordinates

        :returns cost: cost matrix
    """
    
    Nx = x.size
    r = np.ones(Nx)
    Ny = Y[0].size
    vcostcone = np.vectorize(costcone)

    cost = np.exp(-vcostcone(np.tensordot(x,np.ones(Ny),axes =0),
                            np.tensordot(r,np.ones(Ny),axes =0),
                            np.tensordot(np.ones(Nx),Y[0].flatten(),axes =0),
                            np.tensordot(np.ones(Nx),Y[1].flatten(),axes =0),a,b)/sigma)
   
    return cost



def generatecouplingcone(X,x,fundet,a,b,sigma,det=True):
    """Generates L2 distance cost function penalising coupling 
        
       :param X: X[0] flattened meshgrid base space coordinates
                 X[1] \in ]0,M] flattened meshgrid cone coordinates
       :param x: array of base space coordinates          
       :param fundet: fundet[0] function defined on scalar x definining the coupling
                      fundet[1] Jacobian determinant
       :param eps: regularisation parameter (Sinkhorn)
       :param det: if false uses Jac = 1 and vectorizes to allow for generalized Euler solutions

       :returns cost: cost matrix (rectangular Nx*Nr times Nx)
    """   
    
    if det:
        fundetx = fundet(x)
        cost = generatecostcone(X,[fundetx[0],fundetx[1]],a,b,sigma,penalty=10.0)
        return cost 
    else:
        vfun = np.vectorize(fundet)
        funx = vfun(x)
        cost = generatecostcone(X,[funx,np.ones(funx.shape)],a,b,sigma,penalty=10.0)
        return cost


def setcurrentkernelcone(k,K):
    """Returns flag which tells if we are at last time step
       
       :param k: current time step (from 0 to K-1)
       :param K: total number of time steps

       :returns: 0 for k=0, 1 for 1<k<K-1, 2 for k=K-1
    """
    return 0 + (k>0.1) + (k==K-1) 

def liftmultipliercone(p,y):
    """ Maps Lagrange multiplier on base space to one on cone 
 
    :param p: lagrange multiplier (len= Nx)  array
    :param y: coordinates in radial direction
 
    :returns: lagrange multiplier (len= Nx*Nr)
    """
    Nr = len(y)
    return (np.exp(((np.tile(p,(Nr,1)).T)*y)).T).flatten() 


def computepseudomarginalcone(k,K,y,G,PMAT):
    """Computes kth pseudo marginal (needs to be multiplied by kth Lagrange multiplier to produce the kth marginal).
       Used for update of kth Lagrange multiplier
  
       :param k: current time step (from 0 to K-1)
       :param K: total number of time steps
       :param y: array of coordinates in radial direction
       :param G: list G = [Xi0init,Xi0,Xi1]
       :param Xi0init: cost associated with first and second time steps
       :param Xi0: cost associated to successive time steps
       :param Xi1: cost associated to coupling
       :param PMAT: array containing Lagrange multipliers (rows) in log scale to enforce marginals 
    """ 
    sequence = np.roll(range(K),-k-1)
    temp_kernel = G[setcurrentkernelcone(k,K)]

    for ii in sequence[:-1]:
        if ii == 0:
            U = np.exp(PMAT[ii,:])
        else:
            U = liftmultipliercone(PMAT[ii,:],y)
        temp_kernel = (temp_kernel*U).dot(G[setcurrentkernelcone(ii,K)])

    return np.diag(temp_kernel)


def computemultipliercone(p_old,pseudomarg,y,nu):
    """Computes updated Lagrange multiplier (after the first time step) by solving nonlinear equation
       
       :param p_old: old Lagrange multiplier
       :param pseudomarg: pseudo marginal array
       :param y: radial coordinate vector
       :param nu: marginal on base space 
    """
    
    def objective(p,i,pseudomargmat,y,nu):
        """ Computes marginal residual """
        
        #Ny = len(y)
        #newDensity = 0.0
        #for j in range(Ny):
        #    newDensity +=  pseudomargmat[j,i]*np.exp(p*y[j])*y[j]
        newDensity = (np.exp(p*y)*y).dot(pseudomargmat[:,i])

        return np.log(newDensity) -np.log(nu[i])
    
    Nx = len(nu)
    Ny = len(y)
    # IS THIS CORRECT?
    pseudomargmat = pseudomarg.reshape(Ny,Nx)
    p_new=np.array(p_old)
    for i in range(Nx):
        dico = so.root(objective,1.0,args=(i,pseudomargmat,y,nu),method="broyden1")#,tol = 1e-8)
        p_new[i] = dico["x"]
    return p_new




def vcomputemultipliercone(p_old,pseudomarg,y,nu):
    """ (VECTORIZED) Computes updated Lagrange multiplier (after the first time step) by solving nonlinear equation
        In order to use this change the function fixedpointcone

       :param p_old: old Lagrange multiplier
       :param pseudomarg: pseudo marginal array
       :param y: radial coordinate vector
       :param nu: marginal on base space 
    """
    
    def objective(p,pseudomargmat,y,nu):
        """ Computes marginal residual """
       
        lognewDensity = np.log(np.diag((np.exp(np.outer(p,y))*y).dot(pseudomargmat)))
        return lognewDensity -np.log(nu)
    

    Nx = len(nu)
    Ny = len(y)

    pseudomargmat = pseudomarg.reshape(Ny,Nx)
    root = so.root(objective,np.ones(Nx),args=(pseudomargmat,y,nu),method="broyden1")#,tol = 1e-8)
    p_new = root["x"]
      
    return p_new


def fixedpointcone(PMAT,G,y,nu):
   """Fixed point map on Lagrange multipliers for multimarginal problem
       
      :param PMAT: array containing logarithm of Lagrange multipliers (rows) to enforce marginals
      :param nu: marginal to be enforced at each time (on base space)
      :param G: list G = [Xi0init,Xi0,Xi1]
      :param Xi0init: cost associated to first and second time steps
      :param Xi0: cost associated to successive time steps
      :param Xi1: cost associated to coupling
      :param nu: marginal to be enforced at each time
   
      :returns LUMAT: updated LUMAT
      :returns err: marginal deviation from previous iteration at time K/2  
   """
   Nx = PMAT.shape[1] #Number of cells
   K =  PMAT.shape[0] #Number of time steps
 
   # Change of vaiables
   
   temp = computepseudomarginalcone(0,K,y,G,PMAT)   
   PMAT[0,:] = np.log(nu/temp)
   for imod in range(1,K):
       # Each iteration updates the row (time level) imod in UMAT
       temp = computepseudomarginalcone(imod,K,y,G,PMAT)
       Pnew =  computemultipliercone(PMAT[imod,:],temp,y,nu)
       if imod == int(K/2):
            newDensity = 0.0
            U =  liftmultipliercone(PMAT[imod,:],y)
            marg = np.sum((temp*U).reshape(len(y),Nx).T*y,axis=1)
            err = np.sum(np.abs(marg-nu))
            #err = np.sum(np.abs(PMAT[imod,:]))
       PMAT[imod,:] = Pnew


   return PMAT, err


def computetransportcone(PMAT,k_map,y,G, conedensity_flag = False):
    """Compute plan time step 0  -> time step k_map
 
       :param PMAT: array containing Lagrange multipliers (rows) to enforce marginals
       :param k_map: time step (needs to be 0<k_map<K-1) at which computing the coupling
       :param G: list G = [Xiinit, Xi0,Xi1]
       :param Xi0: cost associated to successive time steps
       :param Xi1: cost associated to coupling
       :param conedensity_flag: flag to compute particle density on cone 

       :returns T: transport plan   
       :returns conedensity: particle density on cone
    """
    Nx = PMAT.shape[1]
    K = PMAT.shape[0]
    if k_map == 0:
        # Only works for homogeneous density and 1 in y

        if conedensity_flag:
            return np.outer(y==1,np.ones(Nx)/Nx)
        else:
            return np.eye(Nx)/Nx
    else:
        temp_kernel2 = G[0]
        for ss in range(1,k_map):
             U = liftmultipliercone(PMAT[ss,:],y)
             temp_kernel2 = (temp_kernel2*U).dot(G[1])
        temp_kernel = G[setcurrentkernelcone(k_map,K)]
        for ss in range(k_map+1,K):
              U = liftmultipliercone(PMAT[ss,:],y)
              temp_kernel = (temp_kernel*U).dot(G[setcurrentkernelcone(ss,K)])
            
    U = liftmultipliercone(PMAT[k_map,:],y)
    T_tot = (temp_kernel*np.exp(PMAT[0,:]))*((temp_kernel2*U).T)
    if conedensity_flag:
        return np.sum(T_tot.reshape(len(y),Nx,Nx),axis=1)
    else:
        return np.sum(T_tot.reshape(len(y),Nx,Nx),axis=0)



