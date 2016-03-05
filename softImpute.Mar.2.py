#!/usr/bin/python 

from sklearn import preprocessing
import scipy.sparse as sps
import scipy
from scipy import linalg
import numpy as np
#import matplotlib.pyplot as plt
import time
import random


filename="/Users/Derrick/Desktop/191Winter16/ml-100k/u.data"
#filename="/Users/Derrick/Desktop/191Winter16/ml-1m/ratings.dat"



def generate_training_dataset(filename):
  array=np.genfromtxt(filename,dtype="int")
  population_size=len(array)
  population_indices=np.arange(population_size)
  training_indices=random.sample(population_indices,int(population_size*0.8))
  training_array=array[training_indices]
  np.savetxt("training_dataset",training_array,delimiter="\t")
  
def loadMatrix(filename):
  array=np.genfromtxt(filename)
  userID=array[:,0]
  movieID=array[:,1]
  values=array[:,2]
  X=sps.coo_matrix((values,(userID-1,movieID-1)),shape=(943,1682))
  #X=sps.coo_matrix((values,(userID-1,movieID-1)),shape=(6040,3952))
  #returns a coo_matrix X
  return X

# Algorithm 3.1
def main():
  print "loading matrix X"
  X=loadMatrix(filename)
  m,n=np.shape(X)
  # r is the rank
  r=5
  #Lambda is the regularization parameter
  Lambda=40
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
  
  X=X.todense()
  X=preprocessing.scale(X)
  print "setting threshold"
  threshold=10**(-6)
  iterations=0
  
  # tempx is a copy of the sparse matrix to be used as Xstar
  #tempx=sps.coo_matrix(X)
  #tempx=tempx.todense()
  tempx=X
  t=time.time()
  while(True):
    U_old=U
    V_old=V
    D_squared_old=D_squared
    print "updating B"
    
    B_tilda_t=linalg.solve(D**2+Lambda*np.identity(r),D.dot(U.T.dot(tempx)))
    B_tilda=B_tilda_t.T
    #updating V and D_squared
    V,D_squared,Vt=linalg.svd(B_tilda.dot(D),full_matrices=False)
    D=np.sqrt(D_squared)
    D=np.diagflat(D)
    B=V.dot(D)
    
    ##?????
    U=U.dot(Vt)
    
    
    
    #Updating Xstar
    xhat=U.dot(D**2).dot(V.T)
    tempx[row,col]=xhat[row,col]
    
    # measuring convergence of B
    D_squared=np.diagflat(D_squared)
    denom=np.trace(D_squared_old**2)
    UtU=D_squared.dot((U.T.dot(U_old)))
    VtV=D_squared_old.dot(V_old.T.dot(V))
    UVproduct= np.trace(UtU.dot(VtV))
    numerator=denom+np.trace(D_squared**2)-2*UVproduct
    Delta_B=numerator/max(denom,10**(-100))
    
    #plt.scatter(time.time()-t,Delta_B,c="red")
    
    print "Updating A"
    U_old=U
    V_old=V
    D_squared_old=D_squared
    
    A_tilda=(tempx.dot(V.dot(D))).dot(linalg.inv(D**2+Lambda*np.identity(r)))    
    U,D_squared,Vt=linalg.svd(A_tilda.dot(D),full_matrices=False)
    D=np.sqrt(D_squared)
    D=np.diagflat(D)
    A=U.dot(D)
    
    
    ##???
    V=V.dot(Vt)
    
    
    #Updating Xstar
    xhat=U.dot(D**2).dot(V.T)
    tempx[row,col]=xhat[row,col]
      
      
    # measuring convergence of A
    D_squared=np.diagflat(D_squared)
    denom=np.trace(D_squared_old**2)
    UtU=D_squared.dot((U.T.dot(U_old)))
    VtV=D_squared_old.dot(V_old.T.dot(V))
    UVproduct= np.trace(UtU.dot(VtV))
    numerator=denom+np.trace(D_squared**2)-2*UVproduct
    Delta_A=numerator/max(denom,10**(-100))
    
    
    iterations+=1
    print "number of iterations: " +str(iterations)
    print "Delta_B = " +str(Delta_B)
    print "Delta_A = " +str(Delta_A)
    
    # plotting the convergence rate
    #plt.scatter(time.time()-t,Delta_A,c="blue")
    

    #we break the loop upon convergence
    if(Delta_A<threshold and Delta_B<threshold):
      break

    print "threshold not reached, continue"
    
    
  #plt.title("Regularization Lambda is: "+str(Lambda) +"  rank is: " +str(r))
 # plt.show()
  
  Bt=B.T

  ABt=A.dot(Bt)

  for cood in Omega:
      i,j=cood
      X[i,j] = X[i,j]-ABt[i,j]
      
  Xstar=X+ABt
  M=Xstar.dot(V)
  U,D,Rt=linalg.svd(M,full_matrices=False)
  V=V.dot(Rt.T)
  #threshold the matrix D
  threshold_lambda=40
  D=np.maximum(D-threshold_lambda,0)
  print D
  D=np.diagflat(D)
  
  print "D is:"
  print D
 
  full_matrix=(U.dot(D**2)).dot(V.T)
  #full_matrix=preprocessing.scale(full_matrix)
  np.savetxt("Full Matrix", full_matrix,delimiter=" ",fmt="%d")

if __name__ == '__main__':
  main()
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  