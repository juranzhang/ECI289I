from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

# this data uses "ATT distance" defined here:
# http://debian.cse.msu.edu/CSE/pub/GA/programs/Genocop-doc/TSP/READ_ME
# (this explains the division by 10 inside the square root)
def distance(tour, xy):
  # index with list, and repeat the first city
  tour = np.append(tour, tour[0])
  d = np.diff(xy[tour], axis=0) 
  return np.sqrt((d**2).sum(axis=1)).sum()

def get_random_cities(N):
  return np.random.rand(N,2)
  
num_seeds = 10
ft = np.zeros((5, num_seeds))
T0 = 100 # initial temperature
alpha = 0.95 # cooling parameter
N = [50, 100, 200, 400, 800]

for index in range(5):
    # create city locations
    xy = get_random_cities(N[index])
    num_cities = len(xy)
    
    # simulated annealing
    for seed in range(num_seeds):
      np.random.seed(seed+8)
    
      # random initial tour
      tour = np.arange(num_cities)
      tour = np.random.permutation(tour)
      bestf = distance(tour, xy)
      i=0
      while bestf>np.sqrt(2*N[index]):
        # mutate the tour using two random cities
        trial_tour = np.copy(tour) # do not operate on original list
        a = np.random.randint(num_cities)
        b = np.random.randint(num_cities)
    
        # option 1: just swap the pair (in-class version)
        trial_tour[a],trial_tour[b] = trial_tour[b],trial_tour[a]
    
        trial_f = distance(trial_tour, xy)
        
        if trial_f < bestf:
          tour = trial_tour
          bestf = trial_f
        
        i=i+1
        
      print i
      print bestf
      print np.sqrt(2*num_cities)
      print num_cities
      ft[index,seed] = i

np.savetxt('NFE.txt', ft)




