#!/usr/bin/python 

from scipy.sparse import *
from scipy import *
import re

def main():
  filename="/Users/Derrick/Desktop/191Winter16/ml-1m/test.dat"
  f=open(filename,'r')
  X=dok_matrix((10,10))
  for line in f:
    match=re.search('(\d+)::(\d+)::(\d+)::',line)
    if match:
      userID=int(match.group(1))
      movieID=int(match.group(2))
      rating=int(match.group(3))
      X[userID,movieID]=rating
  
  print X.keys()


if __name__ == '__main__':
  main()