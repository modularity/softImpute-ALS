#!/usr/bin/python 

import scipy.sparse as sps
import scipy
from scipy import linalg
import numpy as np

import rpy2

"""

Implementation of Algorithm 3.1 on page 3375 of "Matrix Completion and Low-Rank SVD via
Fast Alternating Least Squares" by Hastie et al(2015)

"""

# data file
movielens100k="movielens/u.data"

THRESHOLD = 10**(-5)
Omega = None
Omega_t = None
X = None
X_t = None
r = 5

# specify regulation parameter
LAMBDA = 300

# returns a coo_matrix X
def arraytoXmatrix(array):
  global X
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

def init():
  global X, X_t, Omega, Omega_t

  # stores all data in file within an array
  array = np.genfromtxt(movielens100k)

  # returns a coo_matrix X
  print "loading matrix X"
  X = arraytoXmatrix(array)
  m,n = np.shape(X)

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
  return { 'A': A, 'B': B, 'D': D, 'U': U, 'V': V, 'finished': False }

# utilize Hastie et al's softImpute() in R for comparison method
# wrapper from R to python with rp2
def softImputeR(X, r, LAMBDA, type, thresh , maxit, trace.it, warm.start, final.svd): 
"""
softImpute(x, rank.max = 2, lambda = 0, type = c("als", "svd"), thresh = 1e-05,
maxit = 100, trace.it = FALSE, warm.start = NULL, final.svd = TRUE)
x
An m by n matrix with NAs, can be of class "Incomplete"
can hbe centered and scaled via biScale()

rank.max
This restricts the rank of the solution. If sufficiently large, and with
type="svd" the solution solves the nuclear-norm convex matrix-completion problem. 
In this case the number of nonzero singular values returned will be less than or equal to rank.max.
If smaller ranks are used, the solution is not guaranteed to solve the problem, although still results in good local minima.
rank.max should be no bigger than min(dim(x)-1

lambda
nuclear-norm regularization parameter. If lambda=0, algorithm reverts to "hardImpute", 
convergence is typically slower & to local minimum.
Ideally lambda should be chosen so that the solution reached has rank slightly less than rank.max. 
See lambda0() for computing the smallestlambda with a zero solution.

type
two algorithms are implements, type="svd" or the default type="als". 
The "svd" algorithm repeatedly computes the svd of the completed matrix, and soft thresholds  its  singular  values.   
Each  new  soft-thresholded  svd  is  used  to  re-impute the missing entries.  
For large matrices of class "Incomplete", the svd is achieved by an efficient form of alternating orthogonal ridge regression. 
The softImpute "als" algorithm uses this same alternating ridge regression, but updates the imputation at each step, 
leading to quite substantial speedups in some cases.  
The "als" approach does not currently have the same theoretical convergence guarantees as the "svd" approach.

thresh
convergence threshold, measured as the relative change in the Frobenius norm between two successive estimates.

maxit
maximum number of iterations.

trace.it
with trace.it=TRUE, convergence progress is reported.

warm.start
an svd object can be supplied as a warm start.  This is particularly useful when constructing a path of solutions with 
decreasing values of lambda and increasing rank.max.  The previous solution can be provided directly as a warm start for the next.

final.svd
only applicable to type="als".  The alternating ridge-regressions do not lead to exact zeros.  
With the default final.svd=TRUE, at the final iteration, a one step unregularized iteration is performed, 
followed by soft-thresholding of the singular values, leading to hard z
"""

def calculate(A, B, D, U, V):
  global LAMBDA, Omega, Omega_t, X, X_t

  #store once to reuse in (22) and (23)
  DD = D**D
  DDlambda = DD + LAMBDA*np.identity(r)
  U_t = np.transpose(U)
  B_t = np.transpose(B)

  # create sparse plus low-rank matrix analog for X
  tempX = X
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
  
  # updating B for substep 2b
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
  tempX = X_t

  #update from step2 for reuse 
  DD = D**2
  DDlambda = DD + LAMBDA*np.identity(r)
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
  nebla_FA=(np.trace(DD)+np.trace(D**2)-2*np.trace(D.dot(np.transpose(U)).dot(U_new).dot(D)))/np.trace(DD)
   
  # Updating U, D and A=UD
  A = U.dot(D)
  
  print "nebla_FA is:" +str(nebla_FA)
  print "nebla_FB is:" +str(nebla_FB)

  #If both have converged we break the loop
  if (nebla_FA<THRESHOLD and nebla_FB<THRESHOLD):
    return { 'A': A, 'B': B, 'D': D, 'U': U, 'V': V, 'finished': True };

  print "threshold not reached, continue"
  return { 'A': A, 'B': B, 'D': D, 'U': U, 'V': V, 'finished': False };


def main():
  # variables: X, Lambda, Omega, A, B, D, U
  # output: M, U, V, Diag
  state = init();
 

   #iterations count
  iterations=0

  while(state['finished'] == False):
    state = calculate(state['A'], state['B'], state['D'], state['U'], state['V'])
    iterations+=1
    print "number of iterations: " +str(iterations)
  print "finished"

  
  """
  # R code for softImpute-ALS (2015)
  softImputeR(X, r, LAMBDA, type = "als", thresh = 1e-05, maxit = 100, trace.it = FALSE, warm.start = NULL, final.svd = TRUE)

  # R code for softImpute (2010)
  softImputeR(X, r, LAMBDA, type = "svd", thresh = 1e-05, maxit = 100, trace.it = FALSE, warm.start = NULL, final.svd = TRUE)

  """


if __name__ == '__main__':
  main()
