import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = loadtxt("hyp.txt",delimiter=',') 
U,p = stats.mannwhitneyu(data[0],data[1],alternative='greater')
if p<0.05/3:
    print ('Reject the null hypothesis')
else:
    print ('fail to reject')

U,p = stats.mannwhitneyu(data[2],data[1],alternative='greater')
if p<0.05/3:
    print ('Reject the null hypothesis')
else:
    print ('fail to reject')
    
U,p = stats.mannwhitneyu(data[0],data[2],alternative='less')
if p<0.05/3:
    print ('Reject the null hypothesis')
else:
    print ('fail to reject')