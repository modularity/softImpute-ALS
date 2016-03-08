#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import re

partial_filename="/Users/Derrick/Desktop/191Python/plot_data_for_rank_"

def main():
  the_list_of_ranks=[3,5,7,10,12,15,20,25,30,40,60,75,100,125,150,175,200]
  #the_list_of_ranks=[3,5,10,15,20,25,40,75,100,150,200]
  point_radius=5
  np.random.seed(6)
  for rank in the_list_of_ranks:
    point_radius+=2
    f=open(partial_filename+str(rank),'r')
    random_color=np.random.rand(3,1)
    for line in f:
      match=re.search(r'([\d.e+-]+)\s([\d.e+-]+)',line)
      time=float(match.group(1))
      relative_change=float(match.group(2))
      plt.scatter(time,relative_change,c=random_color,alpha=0.68,s=point_radius)
    f.close()
  
  plt.axhline(y=10**(-5),c="green")
  plt.title("lambda = 60 ")
  plt.xlabel("Time in seconds")
  plt.ylabel("Relative change (log scale)")
  plt.ylim(0,10**(-4))
  plt.xlim(xmin=20,xmax=80)
  plt.ticklabel_format(style='sci',axis='y')
  plt.savefig('convergence_plot_Lambda_60.png', bbox_inches='tight',dpi=400)
  plt.show()
  
  
  
  
if __name__=='__main__':
  main()