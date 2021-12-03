import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from numpy import *
from matplotlib.cm import get_cmap
colors = [i for i in get_cmap('tab20').colors]
colors_b = [i for i in get_cmap('tab20b').colors]
colors_c = [i for i in get_cmap('tab20c').colors]
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple


def smooth(a,weight=0.95):
    smoothed = []
    last = a[0]
    for point in a:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed
r=np.loadtxt('/Users/caochenfeng/Downloads/RL-QITE/cut_5.txt')
r=np.array(r)
r1=np.loadtxt('/Users/caochenfeng/Downloads/RL-QITE/qite.txt')
r1=np.array(r1)
m=smooth(r)
m1=smooth(r1)

plt.figure(figsize=(13.8, 4.8))

plt.subplot(1,2,1)
plt.plot(np.arange(len(m)),m, "-", color = colors_b[0], markersize = 1, linewidth=1.5)

plt.xlabel(r'Iteration', fontsize = "xx-large", fontname = "Times New Roman")
plt.ylabel(r'Reward', fontsize = "xx-large", fontname = "Times New Roman")


plt.title("(a)", loc="left", fontsize=18)

# plt.legend(loc = 'best', fontsize = "x-large")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(1,2,2)
plt.plot(np.arange(len(m1)),m1, "-", color = colors_b[0], markersize = 1, linewidth=1.5)

plt.xlabel(r'Iteration', fontsize = "xx-large", fontname = "Times New Roman")
plt.ylabel(r'Reward', fontsize = "xx-large", fontname = "Times New Roman")

plt.title("(b)", loc="left", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("FigC-1.pdf")
plt.close()

