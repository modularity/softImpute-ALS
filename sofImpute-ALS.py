#!/usr/bin/python 

from io import StringIO
import scipy.sparse as sps
import scipy
from scipy import linalg
import numpy as np


#filename="******/ratings.dat"

def loadMatrix(filename):

  array=np.genfromtxt(filename,delimiter="::")
  userID=array[:,0]
  movieID=array[:,1]
  values=array[:,2]
  X=sps.coo_matrix((values,(userID,movieID)),shape=(6041,3953))
  
  #returns a coo_matrix X
  return X

# Algorithm 3.1
def SoIMP_ALS():
  X=loadMatrix(filename)
  m,n=shape(X)
  r=100
  Lambda=50
  # 1.initialize matrix U
  #m>>r

  #U is an mxr matrix
  U = np.random.randn(m, r)
  #Calling QR Factorization on U
  Q,R = linalg.qr(U)
  U=Q[:,0:r]
  #U is now a matrix with orthonormal columns
  
  #D is an rxr identity matrix of type numpy.ndarray
  D=np.identity(r)
  A=U.dot(D)

  #V is nxr
  V=np.zeros(n*r).reshape(n,r)
  #B is an nxr matrix
  B=V.dot(D)
  
  # Now we want to find Btilda transpose
  X=X.todok()
  Xt=X.transpose()
  #Omega is the <'list'> of coordinates with nonzero entries
  Omega=X.keys()
  Omegat=Xt.keys()
  
  # term (22)
  temp=X
  for cood in Omega:
    i,j=cood
    #find the (i,j)-th value of A*B^t
    temp[i,j] = X[i,j]-A[i,:].dot((transpose(B)[:,j]))
    
  temp=temp.tocsr()
  temp=transpose(U)*temp
  temp=D.dot(temp)
  twenty_two=linalg.solve(D*D+Lambda*np.identity(r),temp)
  
  # term (23)
  temp=D.dot(B.transpose())
  temp=D.dot(D)
  twenty_three=linalg.solve(D.dot(D)+Lambda*np.identity(r),temp)
  
  B=transpose(twenty_two+twenty_three)
  
  # part(c) i.e. updating V and D
  
  V,D_squared,Vt = linalg.svd(B.dot(D),full_matrices=False)
  D=np.sqrt(D_squared)
  
  # Step 3
  
  # term (22')
  temp=Xt
  for cood in Omegat:
    i,j=cood
    temp[i,j] = Xt[i,j]-B[i,:].dot((transpose(A)[:,j]))
  
  temp=temp.tocsr()
  temp=transpose(V)*temp
  temp=D.dot(temp)
  twenty_two=linalg.solve(D*D+Lambda*np.identity(r),temp)
  
  # term (23')
  temp=D.dot(A.transpose())
  temp=D.dot(D)
  twenty_three=linalg.solve(D.dot(D)+Lambda*np.identity(r),temp)
  
  A=transpose(twenty_two+twenty_three)
  
  # part (c) updating U and D
  U,D_squared,Vt = linalg.svd(A.dot(D),full_matrices=False)
  D=np.sqrt(D_squared)
  
  
  

  
  
  
  
  
  
  
  
  
  
  
  
  
