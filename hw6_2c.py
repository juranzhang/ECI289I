import numpy as np 
import matplotlib.pyplot as plt

N = np.array([50,100,200,400,800])
N = np.log10(N)
ft = np.loadtxt("NFE.txt")

NFE2converge = np.mean(ft, axis=1)
NFE2converge = np.log10(NFE2converge)
beta = np.polyfit(N, NFE2converge, deg=1)
p = np.poly1d(beta)
xx = np.linspace(1.5,3.2, 16000)
plt.scatter(N,NFE2converge, 50, 'steelblue')
plt.plot(xx, p(xx), color='indianred', linewidth=2)
plt.xlabel('log10 of problem size')
plt.ylabel('log10 of NFE')
plt.show()