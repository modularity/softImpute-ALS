#!/usr/bin/python

import matplotlib.pyplot as plt
import re

Lambda=120
testing_rmse="/Users/Derrick/Desktop/191Python/L_"+str(Lambda)+"/list_testing_rmse"
training_rmse="/Users/Derrick/Desktop/191Python/L_"+str(Lambda)+"/list_training_rmse"

def main():
  f=open(training_rmse,'r')
  for line in f:
    match=re.search(r'([\d.e+-]+)\s([\d.e+-]+)',line)
    rank=float(match.group(1))
    rmse=float(match.group(2))
    testing_dots=plt.scatter(rank,rmse,c="blue",alpha=0.68)
  f.close()
  f=open(testing_rmse,'r')
  for line in f:
    match=re.search(r'([\d.e+-]+)\s([\d.e+-]+)',line)
    rank=float(match.group(1))
    rmse=float(match.group(2))
    training_dots=plt.scatter(rank,rmse,c="green",alpha=0.68)
  
  plt.legend((testing_dots,training_dots),("Testing RMSE","Training RMSE"),
  scatterpoints=1,loc="center right",ncol=1,fontsize=12)
  plt.title("lambda = "+str(Lambda))
  plt.xlabel("rank")
  plt.ylabel("RMSE")
  plt.xlim(xmin=-5)
  plt.savefig("RMSE_plot_Lambda_"+str(Lambda),bbox_inches="tight",dpi=400)
  plt.show()
  
  
  
if __name__=='__main__':
  main()
