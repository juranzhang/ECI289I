# Juran Zhang, 912664699
# ECI289I Homwork2 ex1
# Compare hill climbing and grid search

import numpy as np 
import matplotlib.pyplot as plt

# function to optimize
def peaks(x):
  a = 3*(1-x[0])**2*np.exp(-(x[0]**2) - (x[1]+1)**2)
  b = 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2)
  c = (1/3)*np.exp(-(x[0]+1)**2 - x[1]**2)
  return a - b - c + 6.551 # add this so objective is always positive

ub = 3.0
lb = -3.0

d = 2 # dimension of decision variable space
s = 0.1 # stdev of normal noise
max_NFE = 3000

m = 5
l = 30 # beware "lambda" is a reserved keyword
num_seeds = 10

ft = np.zeros((num_seeds, max_NFE/l))
st = np.zeros((num_seeds, max_NFE/l))

# separate function for mutation
def mutate(x, lb, ub, sigma):
  x_trial = x + np.random.normal(0, sigma, x.size)
  if np.any(x_trial) > ub:
    x_trial = ub
  elif np.any(x_trial) < lb:
    x_trial = lb
  return x_trial

# (mu,lambda) evolution strategy
for seed in range(num_seeds):
  np.random.seed(seed)

  # random initial population (l x d matrix)
  P = np.random.uniform(lb, ub, (l,d))
  f = np.zeros(l) # we'll evaluate them later
  nfe = 0
  f_best, x_best = None, None

  while nfe < max_NFE:

    # evaluate all solutions in the population
    for i,x in enumerate(P):
      f[i] = peaks(x)
      nfe += 1

    # find m best parents, truncation selection
    ix = np.argsort(f)[:m]
    Q = P[ix, :]

    # keep track of best here
    if f_best is None or f[ix[0]] < f_best:
      f_best = f[ix[0]]
      x_best = Q[0,:]

    # then mutate: each parent generates l/m children (integer division)
    child = 0
    num_better_children = 0
    for i,x in enumerate(Q):
      for j in range(int(l/m)):
        
        P[child,:] = mutate(x, lb, ub, s)
        # print peaks(P[child,:])
        # print peaks(x)
        if peaks(P[child,:]) > peaks(x):
          num_better_children += 1
        child += 1

    ft[seed,nfe/l-1] = f_best
    st[seed,nfe/l-1] = s
    print num_better_children
    if num_better_children > l/5:
      s = s*1
    elif num_better_children < l/5:
      s = s*1

  # for each trial print the result (but the traces are saved in ft)
  print x_best
  print f_best


plt.loglog(range(l,max_NFE+1,l), ft.T, color='steelblue', linewidth=1)
plt.loglog(range(l,max_NFE+1,l), st.T, color='indianred', linewidth=1)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()






