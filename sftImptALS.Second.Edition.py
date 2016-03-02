#!/usr/bin/python 

import scipy.sparse as sps
import scipy
from scipy import linalg
import numpy as np
#import matplotlib.pyplot as plt


#filename="/Users/Derrick/Desktop/191Winter16/ml-100k/u.data"
filename="/Users/Derrick/Desktop/191Winter16/ml-1m/ratings.dat"

def loadMatrix(filename):
  array=np.genfromtxt(filename,delimiter="::")
  userID=array[:,0]
  movieID=array[:,1]
  values=array[:,2]
  #X=sps.coo_matrix((values,(userID-1,movieID-1)),shape=(943,1682))
  X=sps.coo_matrix((values,(userID-1,movieID-1)),shape=(6040,3952))
  #returns a coo_matrix X
  return X

# Algorithm 3.1
def main():
  print "loading matrix X"
  X=loadMatrix(filename)
  m,n=np.shape(X)
  r=5
  Lambda=10
  # 1.initialize matrix U
  #m>>r

  print "creating U"
  #U is an mxr matrix
  U = np.random.randn(m, r)
  #Calling QR Factorization on U
  print "QR Factorization"
  Q,R = linalg.qr(U)
  U=Q[:,0:r]
  #U is now a matrix with orthonormal columns
  
  #D is an rxr identity matrix of type numpy.ndarray
  D=np.identity(r)
  D_squared=D**2  
  A=U.dot(D)

  #V is nxr
  V=np.zeros(n*r).reshape(n,r)
  #B is an nxr matrix
  B=V.dot(D)

  print "changing X to dok_matrix"
  
  #X=X.tocsr()
  #Xt=X.transpose()
  #Omega is the <'list'> of coordinates with nonzero entries

  print "obtaining Omega"
  X=X.todok()
  Omega=X.keys()
  row=[o[0] for o in Omega]
  col=[o[1] for o in Omega]
  #Omegat=Xt.keys()
  
  print "setting threshold"
  threshold=10**(-5)
  iterations=0
  
  # tempx is a copy of the sparse matrix to be used as Xstar
  tempx=sps.coo_matrix(X)
  tempx=tempx.todense()
  while(True):
    U_old=U
    V_old=V
    D_squared_old=D_squared
    print "updating B"
    
    B_tilda_t=linalg.solve(D**2+Lambda*np.identity(r),D.dot(U.T*tempx))
    B_tilda=B_tilda_t.T
    #updating V and D_squared
    V,D_squared,Vt=linalg.svd(B_tilda.dot(D),full_matrices=False)
    D=np.sqrt(D_squared)
    D=np.diagflat(D)
    B=V.dot(D)
    #Updating Xstar
    xhat=U.dot(D**2).dot(V.T)
    tempx[row,col]=xhat[row,col]
    
    D_squared=np.diagflat(D_squared)
    denom=np.trace(D_squared_old**2)
    UtU=D_squared.dot((U.T.dot(U_old)))
    VtV=D_squared_old.dot(V_old.T.dot(V))
    UVproduct= np.trace(UtU.dot(VtV))
    numerator=denom+np.trace(D_squared**2)-2*UVproduct
    Delta_B=numerator/max(denom,1e-9)
    
    print "Updating A"
    U_old=U
    V_old=V
    D_squared_old=D_squared
    
    A_tilda=(tempx*V.dot(D)).dot(linalg.inv(D**2+Lambda*np.identity(r)))    
    U,D_squared,Vt=linalg.svd(A_tilda.dot(D),full_matrices=False)
    D=np.sqrt(D_squared)
    D=np.diagflat(D)
    A=U.dot(D)
    
    #Updating Xstar
    xhat=U.dot(D**2).dot(V.T)
    tempx[row,col]=xhat[row,col]
      
      
    # measuring convergence
    D_squared=np.diagflat(D_squared)
    denom=np.trace(D_squared_old**2)
    UtU=D_squared.dot((U.T.dot(U_old)))
    VtV=D_squared_old.dot(V_old.T.dot(V))
    UVproduct= np.trace(UtU.dot(VtV))
    numerator=denom+np.trace(D_squared**2)-2*UVproduct
    Delta_A=numerator/max(denom,1e-9)
    
    iterations+=1
    print "number of iterations: " +str(iterations)
    print "Delta_B = " +str(Delta_B)
    print "Delta_A = " +str(Delta_A)

    #we break the loop upon convergence
    if(Delta_A<threshold and Delta_B<threshold):
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
  
  print "D is:"
  print D
  print "U is:"
  print U
  print "V is:"
  print V 

if __name__ == '__main__':
  main()
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  