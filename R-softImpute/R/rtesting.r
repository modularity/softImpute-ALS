#!/usr/bin/R
source('softImpute.R')

require(methods)
sapply(list.files(pattern="[.]R$", path="R-softImpute/", full.names=TRUE), source)


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


# stores all data in file within an array
array <- read.table(movielens100k, sep="\t", header=FALSE)
#userID = array[:0]
#movieID = array[:1]
#values = array[:2]
#X = sps.coo_matrix((values,(userID-1,movieID-1)),shape=(943,1682))



### here we do it a different way to demonstrate Incomplete
### In practise the observed values are stored in this market-matrix format.
#i = row(xna)[-imiss]
#j = col(xna)[-imiss]
#xnaC=Incomplete(i,j,x=x[-imiss])


library("softImpute")
set.seed(101)
n=200
p=100
J=50
np=n*p
missfrac=0.3
x=matrix(rnorm(n*J),n,J)%*%matrix(rnorm(J*p),J,p)+matrix(rnorm(np),n,p)/5
ix=seq(np)
imiss=sample(ix,np*missfrac,replace=FALSE)
xna=x
xna[imiss]=NA
###uses regular matrix method for matrices with NAs
fit1=softImpute(xna,rank=50,lambda=30)
###uses sparse matrix method for matrices of class "Incomplete"
xnaC=as(xna,"Incomplete")
fit2=softImpute(xnaC,rank=50,lambda=30)
###uses "svd" algorithm
fit3=softImpute(xnaC,rank=50,lambda=30,type="svd")
ximp=complete(xna,fit1)
### first scale xna
xnas=biScale(xna)
fit4=softImpute(xnas,rank=50,lambda=10)
ximp=complete(xna,fit4)
impute(fit4,i=c(1,3,7),j=c(2,5,10))
impute(fit4,i=c(1,3,7),j=c(2,5,10),unscale=FALSE)#ignore scaling and centering



 

# R code for softImpute-ALS (2015)
#softImputeR(X, r, LAMBDA, type = "als", thresh = 1e-05, maxit = 100, trace.it = FALSE, warm.start = NULL, final.svd = TRUE)

# R code for softImpute (2010)
#softImputeR(X, r, LAMBDA, type = "svd", thresh = 1e-05, maxit = 100, trace.it = FALSE, warm.start = NULL, final.svd = TRUE)

  
