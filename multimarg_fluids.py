"""
Module containing classes and functions for computation of a multimarginal problem for fluids 
"""
import os
import time as time
import numpy as np
import scipy.optimize as so
import scipy.misc as misc
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from joblib import Parallel, delayed

##############################################################################################################
# Anderson Accelerator #######################################################################################
##############################################################################################################

### Anderson acceleration scheme as written in 
### Anderson acceleration for fixed point iterations by Walker and Ni.


def AndersonAcceleration2(FixedPointFunc,x,iterations,Lnumber):
    """Anderson acceleration scheme
 
       :param FixedPointFunc: fixed point function returning (x,err) to be accelerated
       :param x: old fixed point estimate as 1d array
       :param iterations: number of Anderson iterations
       :param Lnumber: number of iterations using given fixed point function
       :param errv:  list of errors
    
       :returns x: current fixed point estimate
       :returns errv: list of errors
    """ 

    
    nx = len(x)
    ### Allocate variables
    IteratesMatrixTemp = np.zeros([nx,Lnumber+1])
    DeltaIteratesMatrix = np.zeros([nx,Lnumber])
    
    ResidualsMatrixTemp = np.zeros([nx,Lnumber+1])
    DeltaResidualsMatrix = np.zeros([nx,Lnumber])
    errv =[]
    ### Build Matrices Data on the first Lnumber steps 
    for i in range(Lnumber+1):
        tic= time.time()
        ResidualsMatrixTemp[:,i] = -np.copy(x)
        x,err = FixedPointFunc(x)
        errv.append(err)
        ResidualsMatrixTemp[:,i] += x
        IteratesMatrixTemp[:,i] = np.copy(x)
        toc = time.time() -tic
        print("Memory iteration: %d" %i)
        print("Elapsed time: %f" % toc)
        print("Marginal error: %.15f" %err)   
 
    old_iterate = IteratesMatrixTemp[:,Lnumber]
    old_residual = ResidualsMatrixTemp[:,Lnumber]
    
    DeltaResidualsMatrix = ResidualsMatrixTemp[:,1:] - ResidualsMatrixTemp[:,:-1]
    DeltaIteratesMatrix = IteratesMatrixTemp[:,1:] - IteratesMatrixTemp[:,:-1] 
    
    print("START ANDERSON ITERATIONS")
    ### Iterations of Anderson 
    
    for i in range(iterations):

        tic = time.time()  
        ### compute weights and update current iterate
        weights = np.linalg.lstsq(DeltaResidualsMatrix,old_residual)[0]
        x = old_iterate - DeltaIteratesMatrix.dot(weights)
        #x = old_iterate + old_residual - (DeltaResidualsMatrix + DeltaIteratesMatrix).dot(weights)
         
        new_iterate, err = FixedPointFunc(x)
        errv.append(err)
        ### update matrices
        #index = (i%Lnumber)
        #DeltaIteratesMatrix[:,index] = new_iterate-old_iterate
        #DeltaResidualsMatrix[:,index] = new_iterate-x - old_residual

        DeltaIteratesMatrix = np.roll(DeltaIteratesMatrix,-1,axis=1) 
        DeltaResidualsMatrix = np.roll(DeltaResidualsMatrix,-1,axis=1)
        DeltaIteratesMatrix[:,-1] = new_iterate-old_iterate
        DeltaResidualsMatrix[:,-1] = new_iterate-x - old_residual

        ### update residuals
        old_residual = new_iterate - x
        old_iterate = new_iterate        
        toc = time.time()-tic
        print("Anderson iteration: %d" %i)
        print("Elapsed time: %f" % toc)
        print("Marginal error: %.15f" %err)
    return x, errv

def OptimizationAndersonMixed(FixedPointFunc,P,iterations_simple,iterations_anderson,number_of_steps,memory_number,p):
    Pshape = P.shape
    def functiontemp(x):
        y = np.reshape(x,Pshape)
        z ,err = FixedPointFunc(y,*p)
        return z.flatten(),err
    errv = []
    for t in range(iterations_simple):
        tic = time.time()
        P,err = FixedPointFunc(P,*p)
        errv.append(err)
        toc = time.time() -tic
        print("Simple fixed point iteration: %d" %t)
        print("Elapsed time: %f" % toc)
        print("Marginal error: %f" %err)      
    for r in range(number_of_steps):
        print("MULTISTEP ANDERSON: STEP %d" % r)
      
        P,errva = AndersonAcceleration2(functiontemp,P.flatten(),iterations_anderson,memory_number)        
        P = np.reshape(P,Pshape)
        errv.extend(errva)
    
    return P,errv




##############################################################################################################
# Functions for Euler solutions in 1D ########################################################################
############################################################################################################## 



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

#################################################################################################################
# Functions for CH solutions on cone ############################################################################ 
#################################################################################################################

#    Implementation using m =sqrt(r) change of variable
#    - metric on cone  m g + dm^2/m 
#    - distance on cone m1 + m2 - 2*sqrt(m1 m2) cos( d(x1,x2) wedge pi) 
#    - map (x,m) = (phi,Jac(phi))



def fcone(x):
    return [(np.exp(x)-1.0)/(np.exp(1)-1),np.exp(x)/(np.exp(1)-1)]

def peakon(x):
    if x<=0.5:
       return 1.4*x
    else: 
       return 0.6*x + 0.4

def peakondet(x):
    if x<=0.5:
       return 1.4
    else: 
       return 0.6

def peakon1(x):
    if x<=0.5:
       return 2.0*x
    else: 
       return 1.0

def peakondet1(x):
    if x<=0.5:
       return 2.0
    else: 
       return 0.0



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


def generatecostcone(X,Y,a,b,sigma,penalty=1.0,log_flag=False):
    """ Generate cost using cone metric
      
        :param X: X[0] flattened meshgrid base space coordinates
                  X[1] \in log ]0,M] flattened meshgrid radial cordinates 
        :param Y: Y[0] flattened meshgrid base space coordinates
                  Y[1] \in log ]0,M] flattened meshgrid radial coordinates
        :param a,b: cone metric parameters
        :param sigma: Sinkhorn regularization parameter 
        :param penalty: diminishes sigma, new sigma = sigma/penalty
        :param log_flag: flag if true uses log scale for cost computation
        
        :returns cost: cost matrix
    """
    
    Nx = X[0].size
    Ny = Y[0].size
    vcostcone = np.vectorize(costcone)
    
    cost = -penalty*vcostcone(np.tensordot(X[0].flatten(),np.ones(Ny),axes =0),
                            np.tensordot(X[1].flatten(),np.ones(Ny),axes =0),
                            np.tensordot(np.ones(Nx),Y[0].flatten(),axes =0),
                            np.tensordot(np.ones(Nx),Y[1].flatten(),axes =0),a,b)/sigma
    if log_flag:
        return cost
    else:
        return np.exp(cost)



def generateinitcostcone(x,Y,a,b,sigma,log_flag=False):
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

    cost = -vcostcone(np.tensordot(x,np.ones(Ny),axes =0),
                            np.tensordot(r,np.ones(Ny),axes =0),
                            np.tensordot(np.ones(Nx),Y[0].flatten(),axes =0),
                            np.tensordot(np.ones(Nx),Y[1].flatten(),axes =0),a,b)/sigma
   
    if log_flag:
        return cost
    else:
        return np.exp(cost)

   



def generatecouplingcone(X,x,fun, fundet,a,b,sigma,log_flag=False):
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
    penalty = 40.0 
 
 
    if fundet ==1.0:
        vfun = np.vectorize(fun)
        funx = vfun(x)
        cost = generatecostcone(X,[funx,np.ones(funx.shape)],a,b,sigma,penalty=penalty,log_flag=log_flag)
        return cost
    else: 
        vfun = np.vectorize(fun)
        vfundet = np.vectorize(fundet)  
        funx = vfun(x)   
        fundetx = vfundet(x)
        cost = generatecostcone(X,[funx,fundetx],a,b,sigma,penalty=penalty,log_flag=log_flag)
        return cost     


def setcurrentkernelcone(k,K):
    """Returns flag which tells if we are at last time step
       
       :param k: current time step (from 0 to K-1)
       :param K: total number of time steps

       :returns: 0 for k=0, 1 for 1<k<K-1, 2 for k=K-1
    """
    return 0 + (k>0.1) + (k==K-1) 

def liftmultipliercone(p,y,log_flag=False):
    """ Maps Lagrange multiplier on base space to one on cone 
 
    :param p: lagrange multiplier (len= Nx)  array
    :param y: coordinates in radial direction
 
    :returns: lagrange multiplier (len= Nx*Nr)
    """
    Nr = len(y)
    lift = ((((np.tile(p,(Nr,1)).T)*y)).T).flatten() 
    if log_flag:
        return lift
    else:
        return np.exp(lift)


def computetempkernelcone(temp_kernel,ii,K,y,G,PMAT,log_flag=False):
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

       :returns: pseudomarginal (in log scale if log_flag is on)
    """ 

    if ii == 0:
        if not log_flag:
           U = np.exp(PMAT[ii,:])
        else:
           U =PMAT[ii,:]
    else:
	U = liftmultipliercone(PMAT[ii,:],y,log_flag=log_flag)

    G_loc = G[setcurrentkernelcone(ii,K)]

    if not log_flag:
        temp_kernel = (temp_kernel*U).dot(G_loc)
    else:
	temp_kernel = np.log(np.sum(np.exp(np.expand_dims(temp_kernel+U,axis=2)+np.expand_dims(G_loc,axis=0)), axis=1))
    
 
    return temp_kernel


def computepseudomarginalcone(k,K,y,G,PMAT,log_flag=False):
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

       :returns: pseudomarginal (in log scale if log_flag is on)
    """ 
    sequence = np.roll(range(K),-k-1)
    temp_kernel = G[setcurrentkernelcone(k,K)]

    for ii in sequence[:-1]:
        if ii == 0:
            if not log_flag:
                U = np.exp(PMAT[ii,:])
            else:
                U =PMAT[ii,:]
        else:
            U = liftmultipliercone(PMAT[ii,:],y,log_flag=log_flag)

        G_loc = G[setcurrentkernelcone(ii,K)]
        
        if not log_flag:
            temp_kernel = (temp_kernel*U).dot(G_loc)
        else:
            temp_kernel = np.log(np.sum(np.exp(np.expand_dims(temp_kernel+U,axis=2)+np.expand_dims(G_loc,axis=0)), axis=1)) 
    return np.diag(temp_kernel)


def computemultipliercone(p_old,pseudomarg,y,nu,log_flag=False):
    """Computes updated Lagrange multiplier (after the first time step) by solving nonlinear equation
       
       :param p_old: old Lagrange multiplier
       :param pseudomarg: pseudo marginal array
       :param y: radial coordinate vector
       :param nu: marginal on base space 
    """
     
      
    if log_flag:
         obj = objectiveLOG
    else:
         obj = objective
 
    Nx = len(nu)
    Ny = len(y)
   
    pseudomargmat = pseudomarg.reshape(Ny,Nx)
   
    # Not parallelized version
    p_new=np.array(p_old)
    for i in range(Nx):
        p_new[i] = so.leastsq(obj,1.0,args=(pseudomargmat[:,i],y,nu[i]),xtol = 1e-6)[0]
        #dico = so.root(obj,1.0,args=(pseudomargmat[:,i],y,nu[i]), method="broyden1",tol = 1e-6)
        #dico = so.root(obj,1.0, jac = jacobjective,args=(pseudomargmat[:,i],y,nu[i]),tol = 1e-6)#, method="broyden1")#,tol = 1e-8)
        #p_new[i] = dico["x"]
        #p_new[i] =  so.fsolve(obj,1.0,fprime=jacobjective, args=(pseudomargmat[:,i],y,nu[i]), xtol = 1e-6)
 
    # Parallelized version
    #p_new = np.asarray(Parallel(n_jobs=3)(delayed(optimizer)(pseudomargmati=pseudomargmat[:,i],y=y,nui=nu[i],obj=obj) for i in range(Nx)))
  
    return p_new

def objective(p,pseudomargmati,y,nui):
    """ Computes marginal residual """
    #newDensityLOG = np.log((np.exp(p*y)*y).dot(pseudomargmati))
    newDensityLOG = misc.logsumexp(p*y,b=y*pseudomargmati)
    return newDensityLOG -np.log(nui)

def jacobjective(p,pseudomargmati,y,nui):
    """Computes jacobian of objective """
    A = p*y
    B = y*pseudomargmati
    Jac = np.exp( misc.logsumexp(A,b=B)-misc.logsumexp(A,b =y*B)) +p*pseudomargmati*nui*0.0
    #Jac = ((np.exp(p*y)*y*y).dot(pseudomargmati))/((np.exp(p*y)*y).dot(pseudomargmati)) +p*i*pseudomargmati*nui*0.0
    return Jac 

def objectiveLOG(p,pseudomargmatLOGi,y,nui):
    """ Computes marginal residual using  log coordinates """
    #newDensityLOG =  np.log(np.sum(np.exp( np.log(y) + p*y+pseudomargmatLOGi)))
    newDensityLOG =  misc.logsumexp(np.log(y) + p*y+pseudomargmatLOGi)
    return newDensityLOG-np.log(nui)

 
def optimizer(pseudomargmati,y,nui,obj):
    return so.root(obj,1.0,args=(pseudomargmati,y,nui),method="broyden1",tol=1e-6)['x']


def vcomputemultipliercone(p_old,pseudomarg,y,nu):
    """ (VECTORIZED) Computes updated Lagrange multiplier (after the first time step) by solving nonlinear equation
        In order to use this change the function fixedpointcone

       :param p_old: old Lagrange multiplier
       :param pseudomarg: pseudo marginal array
       :param y: radial coordinate vector
       :param nu: marginal on base space 
    """
    
    def vobjective(p,pseudomargmat,y,nu):
        """ Computes marginal residual """
        #lognewDensity = np.log(np.diag((np.exp(np.outer(p,y))*y).dot(pseudomargmat)))
        lognewDensity = np.log(np.sum((np.exp(np.outer(p,y))*y)*(pseudomargmat).T, axis = 1))
        return lognewDensity -np.log(nu)
    

    Nx = len(nu)
    Ny = len(y)

    pseudomargmat = pseudomarg.reshape(Ny,Nx)
    p_new = so.leastsq(vobjective,np.ones(Nx),args=(pseudomargmat,y,nu),xtol = 1e-6)
         
    return p_new


def fixedpointconerollback(PMATinit,S,G,y,nu):
   """Fixed point map on Lagrange multipliers for multimarginal problem
       
      :param PMAT: array containing logarithm of Lagrange multipliers (rows) to enforce marginals
      :param nu: marginal to be enforced at each time (on base space)
      :param G: list G = [Xi0init,Xi0,Xi1]
      :param Xi0init: cost associated to first and second time steps
      :param Xi0: cost associated to successive time steps
      :param Xi1: cost associated to coupling
      :param nu: marginal to be enforced at each time
   
      :returns PMAT: updated PMAT
      :returns err: marginal deviation from previous iteration at time K/2  
   """
   PMAT = np.copy(PMATinit)
   Nx = PMAT.shape[1] #Number of cells
   K =  PMAT.shape[0] #Number of time steps
   
   # Bacward computation (storing) if S empty list
   if not S:
      S = [G[setcurrentkernelcone(K-2,K)]]*(K-1)
      for ii in range(K-2,0,-1):
         U = liftmultipliercone(PMAT[ii,:],y)
         S[ii-1] = (G[setcurrentkernelcone(ii-1,K)]*U).dot(S[ii])


   # Forward computation
   Uend = liftmultipliercone(PMAT[K-1,:],y)
   temp =  np.sum((S[0]*Uend)*G[2].T,axis=1) 
   PMAT[0,:] = np.log(nu/temp)

   pseudostored = (G[2]*np.exp(PMAT[0,:])).dot(G[0])
   P =[]
   P.append(pseudostored)
   
   for imod in range(1,K):
       print("Computing time step %d of %d ..." %(imod,K))
       # Each iteration updates the row (time level) imod in UMAT
       if imod < K-1:     
            temp = np.sum((S[imod]*Uend)*pseudostored.T,axis=1) 
       else:
            temp = np.diag(pseudostored)
         
       #Pnew = vcomputemultipliercone(PMAT[imod,:],temp,y,nu)
       Pnew =  computemultipliercone(PMAT[imod,:],temp,y,nu)
       
       if imod == int(K/2):
            U =  liftmultipliercone(PMAT[imod,:],y)
            marg = np.sum((temp*U).reshape(len(y),Nx).T*y,axis=1)
            err = np.sum(np.abs(marg-nu))
            
       PMAT[imod,:] = Pnew
       
       if imod< K-1:
            pseudostored = computetempkernelcone(pseudostored,imod,K,y,G,PMAT)
            P.append(pseudostored)
 
   Uend = liftmultipliercone(PMAT[K-1,:],y)
   #Backward computation       
   for ii in range(K-2,0,-1):
       print("Computing time step %d of %d ..." %(ii,K))      
       temp = np.sum((S[ii]*Uend)*P[ii-1].T,axis=1)
       Pnew =  computemultipliercone(PMAT[ii,:],temp,y,nu)
       PMAT[ii,:] = Pnew
       U = liftmultipliercone(PMAT[ii,:],y)
       S[ii-1] = (G[setcurrentkernelcone(ii-1,K)]*U).dot(S[ii])

                 
   return PMAT, err , S



def fixedpointconeroll(PMATinit,G,y,nu,verbose=False):
   """Fixed point map on Lagrange multipliers for multimarginal problem
       
      :param PMAT: array containing logarithm of Lagrange multipliers (rows) to enforce marginals
      :param nu: marginal to be enforced at each time (on base space)
      :param G: list G = [Xi0init,Xi0,Xi1]
      :param Xi0init: cost associated to first and second time steps
      :param Xi0: cost associated to successive time steps
      :param Xi1: cost associated to coupling
      :param nu: marginal to be enforced at each time
   
      :returns PMAT: updated PMAT
      :returns err: marginal deviation from previous iteration at time K/2  
   """
   PMAT = np.copy(PMATinit)
   Nx = PMAT.shape[1] #Number of cells
   K =  PMAT.shape[0] #Number of time steps
   
   # Bacward computation (storing)
   S = [G[setcurrentkernelcone(K-2,K)]]*(K-1)
   for ii in range(K-2,0,-1):
      U = liftmultipliercone(PMAT[ii,:],y)
      S[ii-1] = (G[setcurrentkernelcone(ii-1,K)]*U).dot(S[ii])

   # Forward computation
   Uend = liftmultipliercone(PMAT[K-1,:],y)
   temp =  np.sum((S[0]*Uend)*G[2].T,axis=1) 
   PMAT[0,:] = np.log(nu/temp)

   pseudostored = (G[2]*np.exp(PMAT[0,:])).dot(G[0])
 
   for imod in range(1,K):
       if verbose:
            print("Computing time step %d of %d ..." %(imod,K))
       # Each iteration updates the row (time level) imod in UMAT
       if imod < K-1:     
            temp =np.sum((S[imod]*Uend)*pseudostored.T,axis=1) 
       else:
            temp = np.diag(pseudostored)
        
       if imod == int(K/2):
            U = liftmultipliercone(PMAT[imod,:],y)
            marg = np.sum((temp*U).reshape(len(y),Nx).T*y,axis=1)
            err = np.sum(np.abs(marg-nu))    
       Pnew =  computemultipliercone(PMAT[imod,:],temp,y,nu)
       if imod<K-1:
            U = liftmultipliercone(Pnew,y)
            pseudostored = (pseudostored*U).dot(G[setcurrentkernelcone(imod,K)])
       PMAT[imod,:] = Pnew

   return PMAT, err



def fixedpointcone(PMAT,G,y,nu,log_flag=False):
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
 
   # Change of variables
   
   temp = computepseudomarginalcone(0,K,y,G,PMAT,log_flag=log_flag)   
   if log_flag:
       PMAT[0,:] = np.log(nu)-temp
   else: 
       PMAT[0,:] = np.log(nu/temp)
   for imod in range(1,K):
       print("Computing time step %d of %d ..." %(imod,K))
       # Each iteration updates the row (time level) imod in UMAT
       temp = computepseudomarginalcone(imod,K,y,G,PMAT,log_flag=log_flag)
       #Pnew = vcomputemultipliercone(PMAT[imod,:],temp,y,nu)
       Pnew =  computemultipliercone(PMAT[imod,:],temp,y,nu,log_flag=log_flag)
       if imod == int(K/2):
            newDensity = 0.0
            # NOT using LOG scale here to produce error
            U =  liftmultipliercone(PMAT[imod,:],y)
            if log_flag:
                temp= np.exp(temp)
            marg = np.sum((temp*U).reshape(len(y),Nx).T*y,axis=1)
            err = np.sum(np.abs(marg-nu))
            #err = np.sum(np.abs(PMAT[imod,:]))
       PMAT[imod,:] = Pnew
       

   return PMAT, err


def computetransportcone(PMAT,k_map,y,G, conedensity_flag = False,log_flag = False):
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
    # NOT impplemeneted in log scale: if G in log scale here computes exponential
    if log_flag:
       G =[np.exp(G[0]),np.exp(G[1]),np.exp(G[2])]
   
    if k_map == 0:
        # Only works for homogeneous density and 1 in y
        if conedensity_flag:
            idx = np.argmin(abs(y-1.))
            return np.outer(y==y[idx],np.ones(Nx)/Nx)
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


def savedatacone(errv,PMAT,X1,eps, path):
    """Save data
    
    :param path: folder where to save to 
    """


    # If the directory does not exist, create it
    if os.path.exists(path):
        print("WARNING: Path already exists")
        #return 
    else:
        os.makedirs(path)

    # The final path to save to
    savepath = os.path.join(path, "errv.npy")
    np.save(savepath,errv)
    savepath = os.path.join(path, "PMAT.npy")
    np.save(savepath,PMAT)
    savepath = os.path.join(path, "X1.npy")
    np.save(savepath,X1)
        
    return 
 
def savefigscone(errv,PMAT, X1, eps,G , path , ext='eps',log_flag = False, verbose=True):
    """Save figures 
    
    :param path: folder where to save to 
    """
    # Extract parameters 
    Nx = PMAT.shape[1]
    K = PMAT.shape[0]
    Nr = X1.size
    rmin = X1[0]
    rmax = X1[-1]
    Niter = len(errv)

    # If the directory does not exist, create it
    if os.path.exists(path):
        print("WARNING: Path already exists")
        return 
    else:
        os.makedirs(path)

    if verbose:
       print("Saving figure to '%s'..." % path)

    #Write text file with parameters
    filename = "logfile.py"
    savepath = os.path.join(path, filename)
    f = open(savepath,'w')
    f.write('# PARAMETERS\n')
    f.write("# eps: Sinkhorn regularization parameter\n")
    f.write("# Niter: Number of iterations\n")
    f.write("# K: Number of time steps\n")
    f.write("# Nx Number of points in physical space \n")
    f.write("# Nr Number of points in radial direction\n")
    f.write("# rmax: Min radial bound \n")
    f.write("# rmin: Max radial bound \n\n")
    f.write("eps =%f \nNiter=%d \nK=%d \nNx=%d \nNr=%d \nrmin=%f \nrmax=%f "% (eps,Niter,K,Nx,Nr,rmin ,rmax))

    f.close()


    # Actually save the figures
    filename = "convergence.%s" %ext
    savepath = os.path.join(path, filename)
    fig = plt.semilogy(errv)
    plt.savefig(savepath, format = "eps")
    plt.clf() 	 


    for k_map in range(K):

        Tmap = computetransportcone(PMAT,k_map,X1,G,conedensity_flag = False, log_flag=log_flag)
        fig1 = plt.imshow(-30*Tmap,origin='lower',cmap = 'gray')
	fig1.axes.get_xaxis().set_visible(False)
	fig1.axes.get_yaxis().set_visible(False)
	filename = "transport_%d.%s" % (k_map, ext)
	savepath = os.path.join(path, filename)
	plt.savefig(savepath, format =ext)
        
        Tconemap = computetransportcone(PMAT,k_map,X1,G,conedensity_flag=True, log_flag=log_flag)
	fig1 = plt.imshow(-30*Tconemap,origin='lower',cmap = 'gray')
	fig1.axes.get_xaxis().set_visible(False)
	fig1.axes.get_yaxis().set_visible(False)
	filename = "radialmarg_%d.%s" % (k_map, ext)
	savepath = os.path.join(path, filename)
	plt.savefig(savepath, format =ext )  


    if verbose:
        print("Done")
    
    return 

