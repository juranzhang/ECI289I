import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.loadtxt("hyp.txt",delimiter=',') 

plt.hist([data[0], data[1],data[2]], color=['r','b','g'], alpha=0.5)