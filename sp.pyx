#!/usr/bin/python 

import math
import scipy.sparse as sps
from scipy import linalg
import numpy as np
cimport numpy as np
import time
import random


DTYPE=np.int
ctypedef np.int_t DTYPE_t
ctypedef np.int32_t cINT
ctypedef np.double_t cDOUBLE

filename="/Users/Derrick/Desktop/191Winter16/ml-100k/u.data"
#filename="/Users/Derrick/Desktop/191Winter16/ml-1m/ratings.dat"
testing_file_location="/Users/Derrick/Desktop/191Python/testing_dataset"
training_file_location="/Users/Derrick/Desktop/191Python/training_dataset"


#k is the number of columns in A
#function returns the i-th row of A dotted with the j-th column of B
  
def AomgB(np.ndarray[cDOUBLE,ndim=2] xfill, np.ndarray[cDOUBLE,ndim=2] xhat, np.ndarray[int,ndim=1] row, np.ndarray[int,ndim=1] col, int cardinality, int rank):
  for t in range(cardinality):
    xfill[row[t],col[t]]=xhat[row[t],col[t]]
  return xfill

def generate_training_dataset(filename):
  array=np.genfromtxt(filename,dtype="int")
  population_size=len(array)
  population_indices=np.arange(population_size)
  training_indices=random.sample(population_indices,int(population_size*0.8))
  testing_indices=list(set(population_indices)-set(training_indices))
  training_array=array[training_indices]
  testing_array=array[testing_indices]
  np.savetxt("training_dataset",training_array,delimiter="\t",fmt="%d")
  np.savetxt("testing_dataset",testing_array,delimiter="\t",fmt="%d")
  
def RMSE(U,Dsq,V,file_location):
  Vt=V.T
  array=np.genfromtxt(file_location)
  # m is the number of data points
  m,n=np.shape(array)
  A=U.dot(Dsq)
  userID=array[:,0]
  movieID=array[:,1]
  values=array[:,2]
  squared_error=0
  for i in range(m):
    x=userID[i]-1 
    y=movieID[i]-1
    v=values[i]
    # x-th row of A dot the y-th column of Vt is the predicted value 
    squared_error+=(((A[x,:].dot(Vt[:,y])-v))**2)
  rmse=math.sqrt(squared_error/m)
  return rmse
  
def loadMatrix(filename):
  array=np.genfromtxt(filename)
  userID=array[:,0]
  movieID=array[:,1]
  values=array[:,2]
  X=sps.coo_matrix((values,(userID-1,movieID-1)),shape=(943,1682))
  #X=sps.coo_matrix((values,(userID-1,movieID-1)),shape=(6040,3952))
  #returns a coo_matrix X
  return X

def Frob(D_squared,D_squared_old,U,U_old,V,V_old):
  denom=np.trace(D_squared_old**2)
  UtU=(U_old.T).dot(U)
  UtU=D_squared_old.dot(UtU)
  VtV=D_squared.dot(V.T.dot(V_old))
  UVproduct= np.trace(UtU.dot(VtV))
  numerator=denom+np.trace(D_squared**2)-2*UVproduct
  ratio=numerator/max(denom,10**(-100))
  return ratio


# Algorithm 3.1
def soft_als(training_file_location,rank=5,Lambda=20):
  print "Initializing matrix X."
  X=loadMatrix(training_file_location)

  m,n=np.shape(X)
  # r is the rank
  r=rank
  #Lambda is the regularization parameter
  # 1.initialize matrix U
  print "Creating U."
  #U is an mxr matrix
  U = np.random.randn(m, r)
  #Calling QR Factorization on U
  print "QR factorization."
  Q,R = linalg.qr(U)
  U=Q[:,0:r]
  print "QR factorization complete."
  #U is now a matrix with orthonormal columns
  
  #D is an rxr identity matrix of type numpy.ndarray
  D=np.identity(r)
  Dsq=D**2  
  A=U.dot(D)

  #V is nxr
  V=np.zeros(n*r).reshape(n,r)
  #B is an nxr matrix
  B=V.dot(D)

  print "Transforming X to dok_matrix."
  #Omega is the <'list'> of coordinates with nonzero entries
  X=X.todok()
  print "Getting the keys."
  Omega=X.keys()
  all_keys=[]
  print "Creating all coordinates."
  for i in range(m):
    for j in range(n):
        all_keys.append((i,j))
  print "Created all coordinates."
  all_keys_set=set(all_keys)
  Omega_set=set(Omega)
  complement_set=all_keys_set-Omega_set
  complement=list(complement_set)
  row=[tuple[0] for tuple in complement]
  row=np.asarray(row,dtype=np.int32)
  col=[tuple[1] for tuple in complement]
  col=np.asarray(col,dtype=np.int32)
  cdef int keylength
  keylength=len(row)
  print "Converting X to a dense matrix."
  X=X.todense()
  print "Conversion complete."
  #nz=m*n-sum(xnas)
  xfill = X
  print "Setting threshold."
  cdef double threshold=10e-05
  iterations=0
  print "Recording the initial time."
  t=time.time()
  list_of_convergence=[]
  while(True):
    U_old=U
    V_old=V
    Dsq_old=Dsq
    ## U step
    #B=t(U)%*%xfill
    B=np.dot(U.T, xfill)
    #if(lambda>0)B=B*(Dsq/(Dsq+lambda))
    dsqRatio = np.divide(Dsq,Dsq+Lambda)
    B=dsqRatio.dot(B)
    #Bsvd=svd(t(B))
    u,d,v=linalg.svd(B.T, full_matrices=False)
    #V=Bsvd$u
    V=u
    #Dsq=(Bsvd$d)
    Dsq = np.diagflat(d)
    #U=U%*%Bsvd$v
    U=np.dot(U,v)
    #xhat=U %*%(Dsq*t(V))
    xhat=np.dot(U, Dsq.dot(V.T))
    #xfill[xnas]=xhat[xnas]
    print "Matrix assignment."
    #xfill[row,col]=xhat[row,col]
    xfill=AomgB(xfill,xhat,row,col,keylength,r)
    print "Matrix assignment complete."
    ###The next line we could have done later; this is to match with sparse version
    # if(trace.it) obj=(.5*sum( (xfill-xhat)[!xnas]^2)+lambda*sum(Dsq))/nz
    #obj=(.5*sum((xfill-xhat)[xnas==False]**2)+Lambda*sum(d))/nz
    ## V step
    #A=t(xfill%*%V)
    A=(np.dot(xfill,V)).T
    #if(lambda>0)A=A*(Dsq/(Dsq+lambda))
    A=np.dot(np.divide(Dsq,(Dsq+Lambda)),A)
    #Asvd=svd(t(A))
    u,d,v=linalg.svd(A.T,full_matrices=False)
    U=u
    #U=Asvd$u
    #Dsq=Asvd$d
    Dsq=np.diagflat(d)
    #V=V %*% Asvd$v
    V=V.dot(v)
    #xhat=U %*%(Dsq*t(V))
    xhat=np.dot(U, Dsq.dot(V.T))
    #xfill[xnas]=xhat[xnas]
    print "Matrix assignment."
    #xfill[row,col]=xhat[row,col]
    xfill=AomgB(xfill,xhat,row,col,keylength,r)
    print "Matrix assignment complete."
    ratio=Frob(Dsq,Dsq_old,U,U_old,V,V_old)
    print "Ratio: " +str(ratio)
    #if(trace.it) cat(iter, ":", "obj",format(round(obj,5)),"ratio", ratio, "\n")
    iterations+=1
    print "Number of iterations: " +str(iterations)
    print threshold
    #saving time vs ratio for plotting
    current_time=time.time()-t
    list_of_convergence.append((current_time,ratio))
# plotting the convergence rate
#plt.scatter(time.time()-t,ratio,c="blue")
    #we break the loop upon convergence
    if(ratio < threshold):
      break
    print "Ratio above the threshold. Proceeding to the next iteration.\n"
    
#plt.title("Regularization Lambda is: "+str(Lambda) +"  rank is: " +str(r))
#plt.ylim(0,10**(-2))
#plt.show()
  print "Iterations complete.\n"
  U=xfill.dot(V)
  u,d,v=linalg.svd(U,full_matrices=False)
  U=u
  Dsq=d
  V=V.dot(v)
  Dsq=np.maximum(Dsq-Lambda,0)
  Dsq=np.diagflat(Dsq)
  #full_matrix=U.dot(Dsq).dot(V.T)
  print "Saving U, Dsq, V to files. \n"
  #np.savetxt("Full Matrix_"+str(r), full_matrix,delimiter=" ",fmt="%d")
  np.savetxt("U_"+str(r), U,delimiter=" ")
  np.savetxt("Dsq_"+str(r), Dsq,delimiter=" ")
  np.savetxt("V_"+str(r), V,delimiter=" ")
  np.savetxt("plot_data_for_rank_"+str(r),list_of_convergence,delimiter=" ")
  print "Total elapsed time during iterations for rank "+str(r)+" is "+str(time.time()-t)+" seconds."
  print "Current time stamp: "+str(time.ctime())+"\n"


def main():

  #the_list_of_ranks=[3,5,7,10,12,15,20,25,30,40,60,75,100,125,150,175,200]
  the_list_of_ranks=[3,5,7,10]
  # a list of (rank,rmse) pairs
  list_of_testing_rmses=[]
  list_of_training_rmses=[]
  for rank in the_list_of_ranks:
    print "Rank is "+str(rank)
    soft_als(training_file_location,rank)
  for rank in the_list_of_ranks:
    print "Calculating rmse for rank "+str(rank)
    U=np.genfromtxt("U_"+str(rank))
    Dsq=np.genfromtxt("Dsq_"+str(rank))
    V=np.genfromtxt("V_"+str(rank))
    print "Getting testing rmse."
    testing_rmse=RMSE(U,Dsq,V,testing_file_location)
    list_of_testing_rmses.append((rank,testing_rmse))
    print "getting training rmse."
    training_rmse=RMSE(U,Dsq,V,training_file_location)
    list_of_training_rmses.append((rank,training_rmse))
    
  print "Saving the dictionary of rmses."
  np.savetxt("list_testing_rmse",list_of_testing_rmses,delimiter=" ")
  np.savetxt("list_training_rmse",list_of_training_rmses,delimiter=" ")
  print "list_of_testing_rmses"
  print list_of_testing_rmses
  print "list_of_training_rmses"
  print list_of_training_rmses


if __name__ == '__main__':
  main()
  

  
  
  
  
  
  
  
  
  
  
  
  
  
  