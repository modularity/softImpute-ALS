#!/usr/bin/python 

import scipy.sparse as sps
import scipy
from scipy import linalg
import numpy as np

"""

Implementation of Algorithm 3.1 on page 3375 of "Matrix Completion and Low-Rank SVD via
Fast Alternating Least Squares" by Hastie et al(2015)

"""

## filename="*******/ml-100k/u.data"
# data file
movielens100k="movielens/u.data"

# returns a coo_matrix X
def arraytoXmatrix(array):
  userID = array[:,0]
  movieID = array[:,1]
  values = array[:,2]
  X = sps.coo_matrix((values,(userID-1,movieID-1)),shape=(943,1682))
  return X

# convert observed matrix X to dok and transpose
def dokNtranspose(X):
  print "changing X to dok_matrix"
  X = X.todok()
  X_t = X.transpose()
  return X,X_t

# Omega is the <'list'> of coordinates with nonzero entries
def initOmega(X, X_t):
  print "obtaining Omega"
  Omega = X.keys()
  Omega_t = X_t.keys()
  return Omega, Omega_t

def main():

  # stores all data in file within an array
  array = np.genfromtxt(movielens100k)

  # returns a coo_matrix X
  print "loading matrix X"
  X = arraytoXmatrix(array)
  m,n = np.shape(X)
  r = 5

  # specify regulation parameter
  Lambda = 300

  """ 
  Step 1: initialize matrix U as a random matrix
  U becomes a matrix with orthonormal columns
  Initialize D = I(r), B = VD with V = 0
  """
  #U is an mxr matrix, m>>r  
  print "creating U"
  U = np.random.randn(m, r)

  # Calling QR Factorization on U
  print "QR Factorization"
  Q,R = linalg.qr(U)
  U = Q[:,0:r]
 
  # D is an rxr identity matrix of type numpy.ndarray
  D = np.identity(r)

  """ 
  Step 2: initialize A = UD, B = VD and solve term (20) to update B.
  This updating process has 3 substeps: a, b, c. 
  Note - the iterative component of this algorithm repeats steps 2 and 3.
  """
  # A = UD
  A = U.dot(D)

  # V is an nxr matrix
  V = np.zeros(n*r).reshape(n,r)
  
  # B is an nxr matrix
  B = V.dot(D)

  #convert observed matrix X to dok and transpose
  X, X_t = dokNtranspose(X)
  
  #Omega is the <'list'> of coordinates with nonzero entries
  Omega, Omega_t = initOmega(X, X_t)
  
  print "setting threshold"
  threshold = 10**(-5)

  # variables: X, Lambda, Omega, A, B, D, U
  # updating B for substep 2b
  while(True):

    #store once to reuse in (22) and (23)
    DD = D**2  #** is matrix raised to a certain power I think if you want to form DD it should be D**2
    DDlambda = DD + Lambda*np.identity(r)
    U_t = np.transpose(U)
    B_t = np.transpose(B)

    # create sparse plus low-rank matrix analog for X
    tempX = sps.dok_matrix(X)
    print "iterating over Omega : B"
    for cood in Omega:
      i,j = cood
      #find the (i,j)th value of A*B^t
      tempX[i,j] = X[i,j] - A[i,:].dot((B_t[:,j]))
    print "finished iteration"

    # term (22) in Algorithm 3.1
    tempX = tempX.tocsr()
    tempX = (U_t)*tempX
    tempX = D.dot(tempX)
    twenty_two = linalg.solve( DDlambda, tempX )
    
    # term (23) in Algorithm 3.1
    tempX = (DD).dot(B_t)
    twenty_three = linalg.solve( DDlambda, tempX )
    
    # sum (22) and (23) to update B
    B = np.transpose(twenty_two+twenty_three)
    
    # part(c) i.e. updating V and D
    V_new,D_squared,V_t = linalg.svd(B.dot(D),full_matrices=False)
    D_new = np.sqrt(D_squared)
    D_new = np.diagflat(D_new)

    # checking for convergence of B using (19) on page 3372
    nebla_FB=(np.trace(D**2)+np.trace(D_new**2)-2*np.trace(D.dot(np.transpose(V)).dot(V_new).dot(D_new)))/np.trace(D**2)
  
    #updating V, D and B=VD
    V = V_new
    D = D_new
    B = V.dot(D)

    """ 
    Step 3: 
    Note - the iterative component of this algorithm repeats steps 2 and 3.
    """
    
    # term (22')
    # Updating A
    print "Updating A"
    tempX = sps.dok_matrix(X_t)

    #update from step2 for reuse 
    DD = D**2
    DDlambda = DD + Lambda*np.identity(r)
    A_t = np.transpose(A)

    print "iterating over Omega : A"
    for cood in Omega_t:
      i,j = cood
      tempX[i,j] = X_t[i,j]-B[i,:].dot((A_t[:,j]))

    tempX = tempX.tocsr()
    tempX = (np.transpose(V))*tempX
    tempX = D.dot(tempX)
    twenty_two = linalg.solve(DDlambda,tempX)

    # term (23')D_
    tempX = D.dot(A_t)
    tempX = D.dot(tempX)
    twenty_three = linalg.solve(DDlambda,tempX)   
    
    A = np.transpose(twenty_two+twenty_three)
    
    # part (c) updating U and D
    U_new,D_squared,V_t = linalg.svd(A.dot(D),full_matrices=False)
    D = np.diagflat(np.sqrt(D_squared))
    
    # Checking for convergence of A
    nebla_FA=(np.trace(D**2)+np.trace(D_new**2)-2*np.trace(D.dot(np.transpose(U)).dot(U_new).dot(D_new)))/np.trace(D**2)
     
    # Updating U, D and A=UD
    A = U.dot(D)
    
    print "nebla_FA is:" +str(nebla_FA)
    print "nebla_FB is:" +str(nebla_FB)

    #If both have converged we break the loop
    if (nebla_FA<threshold and nebla_FB<threshold):
      break

    print "threshold not reached, continue"
    
  Bt=B.T
  ABt=A.dot(Bt)
  for cood in Omega:
      i,j=cood
      X[i,j] = X[i,j]-A[i,:].dot((Bt[:,j]))
      
  X=X.todense()
  Xstar=X+ABt
  M=Xstar.dot(V)
  U,D,Rt=linalg.svd(M,full_matrices=False)
  V=V.dot(Rt.T)
  D=np.diagflat(D)

if __name__ == '__main__':
  main()
