#!/usr/bin/python 

import math
import scipy.sparse as sps
from scipy import linalg
import numpy as np
#import matplotlib.pyplot as plt
import time
import random

# data file
filename="movielens/u.data"
#filename="/Users/Derrick/Desktop/191Winter16/ml-1m/ratings.dat"
testing_file_location="testing_dataset"
training_file_location="training_dataset"
def generate_training_dataset(filename):
  array=np.genfromtxt(filename,dtype="int")
  population_size=len(array)
  population_indices=np.arange(population_size)
  training_indices=random.sample(population_indices,int(population_size*0.8))
  testing_indices=list(set(population_indices)-set(training_indices))
  training_array=array[training_indices]
  testing_array=array[testing_indices]
  np.savetxt("training_dataset",training_array,delimiter="\t",fmt="%d")
  np.savetxt("testing_dataset",training_array,delimiter="\t",fmt="%d")
  
def RMSE(U,Dsq,V,file_location):
  Vt=V.T
  array=np.genfromtxt(file_location)
  # m is the number of data points
  m,n=np.shape(array)
  print "m is "+str(m)
  print "n is "+str(m)
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
  UtU=(U.T).dot(U_old)
  UtU=D_squared.dot(UtU)
  VtV=D_squared_old.dot(V_old.T.dot(V))
  UVproduct= np.trace(UtU.dot(VtV))
  numerator=denom+np.trace(D_squared**2)-2*UVproduct
  ratio=numerator/max(denom,10**(-100))
  return ratio

# Algorithm 3.1
def soft_als(training_file_location,rank=5,Lambda=1):
  print "loading matrix X"
  X=loadMatrix(training_file_location)
  print("load data from movielens100k")
  m,n=np.shape(X)
  # r is the rank
  r=rank
  #Lambda is the regularization parameter
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
  Dsq=D**2  
  A=U.dot(D)

  #V is nxr
  V=np.zeros(n*r).reshape(n,r)
  #B is an nxr matrix
  B=V.dot(D)

  print "changing X to dok_matrix"
  #Omega is the <'list'> of coordinates with nonzero entries
  X=X.todok()
  print "getting the keys"
  Omega=X.keys()
  print "finished getting the keys"
  all_keys=[]
  for i in range(m):
    for j in range(n):
        all_keys.append((i,j))
  all_keys_set=set(all_keys)
  Omega_set=set(Omega)
  complement_set=all_keys_set-Omega_set
  complement=list(complement_set)
  row=[tuple[0] for tuple in complement]
  col=[tuple[1] for tuple in complement]
  
  X=X.todense()
  print("X todense")

  #nz=m*n-sum(xnas)
  xfill = X

  print "setting threshold"
  threshold=10**(-5)
  iterations=0
  
  t=time.time()
  list_of_convergence=[]
  print(X)
  while(True):
    U_old=U
    V_old=V
    Dsq_old=Dsq

    ## U step

    #B=t(U)%*%xfill
    B=np.dot(U.T, xfill)
    #if(lambda>0)B=B*(Dsq/(Dsq+lambda))
    dsqRatio = np.divide(Dsq,Dsq+Lambda)
    B=np.dot(dsqRatio, B)
    #Bsvd=svd(t(B))
    u,d,v=linalg.svd(B.T, full_matrices=False)
    #V=Bsvd$u
    V=u
    #Dsq=(Bsvd$d)
    Dsq = np.diagflat(d)
    #U=U%*%Bsvd$v
    U=np.dot(U,v)
    #xhat=U %*%(Dsq*t(V))
    print "forming xhat"
    xhat=np.dot(U, Dsq.dot(V.T))
    #xfill[xnas]=xhat[xnas]
    print "matrix assignment"
    xfill[row,col]=xhat[row,col]
    print "finished updating xfill"
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
    print "matrix assignment"
    xfill[row,col]=xhat[row,col]
    print "finished updating xfill"
    ratio=Frob(Dsq,Dsq_old,U,U_old,V,V_old)
    print "ratio is: " +str(ratio)
    #if(trace.it) cat(iter, ":", "obj",format(round(obj,5)),"ratio", ratio, "\n")
    
    iterations+=1
    print "number of iterations: " +str(iterations)+"\n"
    #saving time vs ratio for plotting
    current_time=time.time()-t
    list_of_convergence.append((current_time,ratio))
# plotting the convergence rate
#plt.scatter(time.time()-t,ratio,c="blue")
    
    #we break the loop upon convergence
    if(ratio < threshold):
      break
    print "threshold not reached, continue"
    
#plt.title("Regularization Lambda is: "+str(Lambda) +"  rank is: " +str(r))
#plt.ylim(0,10**(-2))
#plt.show()

  U=xfill.dot(V)
  u,d,v=linalg.svd(U,full_matrices=False)
  U=u
  Dsq=d
  
  V=V.dot(v)
  threshold_lambda=0
  
  Dsq=np.diagflat(Dsq)
  print "Dsq after svd:"
  print Dsq
  #full_matrix=U.dot(Dsq).dot(V.T)
  print "saving U, Dsq, V to files \n"
  #np.savetxt("Full Matrix_"+str(r), full_matrix,delimiter=" ",fmt="%d")
  np.savetxt("U_"+str(r), U,delimiter=" ")
  np.savetxt("Dsq_"+str(r), Dsq,delimiter=" ")
  np.savetxt("V_"+str(r), V,delimiter=" ")
  np.savetxt("plot_data_for_rank_"+str(r),list_of_convergence,delimiter=" ")


def main():
  the_list_of_ranks=np.arange(5,11,10)
  # a list of (rank,rmse) pairs
  list_of_testing_rmses=[]
  list_of_training_rmses=[]
  for rank in the_list_of_ranks:
    print "rank is "+str(rank)
    soft_als(training_file_location,rank)
  for rank in the_list_of_ranks:
    print "calculating rmse for rank "+str(rank)
    U=np.genfromtxt("U_"+str(rank))
    Dsq=np.genfromtxt("Dsq_"+str(rank))
    V=np.genfromtxt("V_"+str(rank))
    print "getting testing rmse"
    testing_rmse=RMSE(U,Dsq,V,testing_file_location)
    list_of_testing_rmses.append((rank,testing_rmse))
    print "getting training rmse"
    training_rmse=RMSE(U,Dsq,V,training_file_location)
    list_of_training_rmses.append((rank,training_rmse))
    
  print "saving the dictionary of rmses"
  np.savetxt("dict_testing_rmse",list_of_testing_rmses,delimiter=" ")
  np.savetxt("dict_training_rmse",list_of_training_rmses,delimiter=" ")
  print "list_of_testing_rmses"
  print list_of_testing _rmses
  print "list_of_training_rmses"
  print list_of_training_rmses


if __name__ == '__main__':
  main()
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
