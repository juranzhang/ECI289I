import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from matplotlib import animation
from pandas.tools.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D
# function to optimize
# from 
# http://www.mathworks.com/help/gads/using-gamultiobj.html?refresh=true
def mymulti1(x):
  f1 = x[0]**4 - 10*x[0]**2+x[0]*x[1] + x[1]**4 -(x[0]**2)*(x[1]**2);
  f2 = x[1]**4 - (x[0]**2)*(x[1]**2) + x[0]**4 + x[0]*x[1];
  return np.array([f1,f2])

ub = 5
lb = -5

d = 2 # dimension of decision variable space
num_obj = 2
s = 0.02 # stdev of normal noise (if this is too big, it's just random search!)

m = 18
l = 100 # beware "lambda" is a reserved keyword
max_gen = 100 # should be a multiple

# ASSUMES MINIMIZATION
# a dominates b if it is <= in all objectives and < in at least one
def dominates(a,b):
  return (np.all(a <= b) and np.any(a < b))

# select 1 parent from population P
# (Luke Algorithm 99 p.138)
def binary_tournament(P,f):
  ix = np.random.randint(0,P.shape[0],2)
  a,b = f[ix[0]], f[ix[1]]
  if dominates(a,b):
    return P[ix[0]]
  elif dominates(b,a):
    return P[ix[1]]
  else:
    return P[ix[0]] if np.random.rand() < 0.5 else P[ix[1]]

def mutate(x, lb, ub, sigma):
  x_trial = x + np.random.normal(0, sigma, x.size)
  while np.any((x_trial > ub) | (x_trial < lb)):
    x_trial = x + np.random.normal(0, sigma, x.size)
  return x_trial

# assumes minimization
def archive_sort(A, fA, P, fP):

  for i,x in enumerate(P):
    
    dominated = False
    added = False
    to_be_deleted = []
    for j,xA in enumerate(A):
        
        # if population member dominates archive member, replace or delete if 
        # pop has previously been pushed
        if dominates(fP[i,:], fA[j,:]):
            if added == False:
                A[j,:] = P[i,:]
                fA[j,:] = fP[i,:]
                added = True
            else:
                to_be_deleted.append(j)

        # if it's dominated, ignore pop
        elif dominates(fA[j,:], fP[i,:]):
            dominated = True
            break
    
    for k,kA in enumerate(to_be_deleted):
        A = np.delete(A,k,0)
        fA = np.delete(fA,k,0)
    if not dominated and not added:
        A = np.vstack((A,x))
        fA = np.vstack((fA,fP[i,:]))

  return (A,fA)

# a simple multiobjective version of ES (sort of)
np.random.seed(2)

# random initial population (l x d matrix)
P = np.random.uniform(lb, ub, (l,d))
f = np.zeros((l,num_obj)) # we'll evaluate them later
gen = 0
P_save = []
f_save = []
A_save = []
Af_save = []

# archive starts with 2 bad made-up solutions
A = np.zeros_like(P[0:2,:])
fA = 10**10*np.ones_like(f[0:2,:])

while gen < max_gen:

  # evaluate all solutions in the population
  for i,x in enumerate(P):
    f[i,:] = mymulti1(x)

  A,fA = archive_sort(A, fA, P, f)

  # find m parents from nondomination tournaments
  Q = np.zeros((m,d))
  for i in range(m):
    Q[i,:] = binary_tournament(P,f)

  # then mutate: each parent generates l/m children (integer division)
  child = 0
  for i,x in enumerate(Q):
    for j in range(int(l/m)):
      P[child,:] = mutate(x, lb, ub, s)
      child += 1

  gen += 1
  print gen
  
print P
# plt.plot(P[:,0],P[:,1],'ro')
plt.plot(fA[:,0],fA[:,1],'ko')


plt.show()


# anim = animation.FuncAnimation(fig, animate, frames=len(P_save))
# anim.save('mymulti1b.gif', writer='imagemagick')

