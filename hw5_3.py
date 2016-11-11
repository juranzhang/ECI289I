from platypus.core import Problem
from platypus.algorithms import NSGAII
from platypus.types import Real
import pandas
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D
    # objective function for HW 5, from Liao et al. (2007)
# this is a regression approximation of a finite element model
def carcrash(t):
  mass = 1640.2823 + 2.3573285*t[0] + 2.3220035*t[1] + 4.5688768*t[2] + 7.7213633*t[3] + 4.4559504*t[4]
  ain = 6.5856 + 1.15*t[0] - 1.0427*t[1] + 0.9738*t[2] + 0.8364*t[3] - 0.3695*t[0]*t[3] \
        + 0.0861*t[0]*t[4] + 0.3628*t[1]*t[3] - 0.1106*t[0]**2 - 0.3437*t[2]**2 + 0.1764*t[3]**2
  intrusion = -0.0551 + 0.0181*t[0] + 0.1024*t[1] + 0.0421*t[2] - 0.0073*t[0]*t[1] + 0.024*t[1]*t[2] \
              - 0.0118*t[1]*t[3] - 0.0204*t[2]*t[3] - 0.008*t[2]*t[4] - 0.0241*t[1]**2 + 0.0109*t[3]**2


  return [mass, ain, intrusion]


problem = Problem(5, 3)
problem.types[:] = Real(-10, 10)
problem.function = carcrash

algorithm = NSGAII(problem)
algorithm.run(10000)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
data = []
for solution in algorithm.result:
    data.append(solution.objectives)
    ax.scatter(solution.objectives[0], solution.objectives[1], solution.objectives[2],c='r', marker='o')
    print solution.objectives
fig2 = plt.figure(2)
for i in data:
    if i[2] < 0.5:
        plt.plot(i,color = 'b',zorder = 2)
    else:
        plt.plot(i,color = 'g',zorder = 2)
plt.show()