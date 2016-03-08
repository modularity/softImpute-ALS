#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

testing_rmse_partial_address="/Users/Derrick/Desktop/191Python/3D_data/list_testing_rmse_"

def main():
  the_list_of_lambdas=[15,20,40,60,90,120]
  the_list_of_ranks=[3,5,7,10,12,15,20,25,30,40,60,75,100,125,150,175,200]
  np.random.seed(6)
  fig=plt.figure()
  ax=Axes3D(fig)
  for l in the_list_of_lambdas:
    f=open(testing_rmse_partial_address+str(l),'r')
    random_color=np.random.rand(3,1)
    for line in f:
      match=re.search(r'([\d.e+-]+)\s([\d.e+-]+)',line)
      rank=float(match.group(1))
      rmse=float(match.group(2))
      ax.scatter(l,rank,rmse,c=random_color,alpha=0.9,s=8)
    f.close()
 
  ax.set_xlabel("lambda")
  ax.set_ylabel("rank")
  ax.set_zlabel("RMSE")
  ax.set_xlim(xmin=0)
  ax.set_ylim(ymin=0)
  ax.set_title("RMSE with respect to lambda and rank\n elevation=0  azimuth=180")
  ax.view_init(0,180)
    
  
  plt.savefig('3D_0_180.png', bbox_inches='tight',dpi=500)
  plt.show()
  
  
  
  
if __name__=='__main__':
  main()