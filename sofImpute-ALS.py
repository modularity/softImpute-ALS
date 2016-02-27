#!/usr/bin/python 

import scipy.sparse as sps
import scipy
from scipy import linalg
import numpy as np


filename="/Users/Derrick/Desktop/191Winter16/ml-100k/u.data"

def loadMatrix(filename):
  array=np.genfromtxt(filename)
  userID=array[:,0]
  movieID=array[:,1]
  values=array[:,2]
  X=sps.coo_matrix((values,(userID-1,movieID-1)),shape=(943,1682))
  #returns a coo_matrix X
  return X

# Algorithm 3.1
def main():
  print "loading matrix X"
  X=loadMatrix(filename)
  m,n=np.shape(X)
  r=15
  Lambda=50
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
  A=U.dot(D)

  #V is nxr
  V=np.zeros(n*r).reshape(n,r)
  #B is an nxr matrix
  B=V.dot(D)

  print "changing X to dok_matrix"
  
  X=X.todok()
  Xt=X.transpose()
  #Omega is the <'list'> of coordinates with nonzero entries

  print "obtaining Omega"
  Omega=X.keys()
  Omegat=Xt.keys()
  
  print "setting threshold"
  threshold=10**(-5)
  while(True):
    print "updating B"
    # term (22)
    temp=X
    print "iterating over Omega"
    for cood in Omega:
      i,j=cood
      #find the (i,j)-th value of A*B^t
      temp[i,j] = X[i,j]-A[i,:].dot((np.transpose(B)[:,j]))
      
    print "finished iteration"
    temp=temp.tocsr()
    temp=(np.transpose(U))*temp
    temp=D.dot(temp)
    twenty_two=linalg.solve(D*D+Lambda*np.identity(r),temp)
    
    # term (23)
    temp=(D**2).dot(B.transpose())
    twenty_three=linalg.solve(D.dot(D)+Lambda*np.identity(r),temp)
    
    B=np.transpose(twenty_two+twenty_three)
    
    # part(c) i.e. updating V and D
    
    V_new,D_squared,Vt = linalg.svd(B.dot(D),full_matrices=False)
    D_new=np.sqrt(D_squared)
    D_new=np.diagflat(D_new)

    # checking for convergence of B using (19) on page 3372
    nebla_FB=(np.trace(D**2)+np.trace(D_new**2)-2*np.trace(D.dot(np.transpose(V)).dot(V_new).dot(D_new)))/np.trace(D**2)

    #updating V, D and B=VD
    V=V_new
    D=D_new
    B=V.dot(D)

    
    # Step 3
    
    # term (22')
    print "Updating A"
    temp=Xt

    print "iterating over Omega"
    for cood in Omegat:
      i,j=cood
      temp[i,j] = Xt[i,j]-B[i,:].dot((np.transpose(A)[:,j]))
    
    print "finished iteration"
    temp=temp.tocsr()
    temp=(np.transpose(V))*temp
    temp=D.dot(temp)
    twenty_two=linalg.solve(D*D+Lambda*np.identity(r),temp)
    
    # term (23')
    temp=D.dot(A.transpose())
    temp=D.dot(temp)
    twenty_three=linalg.solve(D.dot(D)+Lambda*np.identity(r),temp)
    
    A=np.transpose(twenty_two+twenty_three)
    
    # part (c) updating U and D
    U_new,D_squared,Vt = linalg.svd(A.dot(D),full_matrices=False)
    D_new=np.sqrt(D_squared)
    D_new=np.diagflat(D_new)
    
    # Checking for convergence of A
    nebla_FA=(np.trace(D**2)+np.trace(D_new**2)-2*np.trace(D.dot(np.transpose(U)).dot(U_new).dot(D_new)))/np.trace(D**2)
    
    # Updating U, D and A=UD
    U=U_new
    D=D_new
    A=U.dot(D)
    
    print "nebla_FA is:" +str(nebla_FA)
    print "nebla_FB is:" +str(nebla_FB)

    #If both have converged we break the loop
    if(nebla_FA<threshold and nebla_FB<threshold):
      break

    print "threshold not reached, continue"


if __name__ == '__main__':
  main()
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  