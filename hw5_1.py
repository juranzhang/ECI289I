# read and plot
import numpy as np 
import matplotlib.pyplot as plt

def checker(data,best):
	if data[0]<best[0] and data[1]<best[1]:
		return 1
	elif data[0]>best[0] and data[1]>best[1]:
		return 0
	else:
		return 2

data = np.loadtxt('circle-points.txt')
# data = np.random.random((408, 1600))
best = data
print best
for y in data:
	print y
	for j,x in enumerate(best):
		#print x
		if checker(y,x) == 1:
			best[j] = y

data = np.loadtxt('circle-points.txt')
plt.plot(data[:,0],data[:,1], 'go')
plt.plot(best[:,0],best[:,1], 'ro')
plt.show()