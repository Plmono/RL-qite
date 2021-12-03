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

N_list = [2,3,4,5,6,7,8]
GSE = [-2.2360679774997894, -3.4939592074349344, -4.758770483143625, -6.02667418333227, -7.296229810558769, -8.566772233505588, -9.837951447459464]
D2 = [-2.2221695542224946, -3.35386584547287, -4.305780727338762, -5.294465561801673, -5.751619085906237, -6.347050697137958, -6.893234950666947]
D_rdm= [-2.235884481787055, -3.459844258380183, -4.376302394535666, -5.562063229676033, -6.020212934071477, -6.90915498213478, -7.942137000434444]
Drl = [-2.236063005602385, -3.489494610104148, -4.739892815152563, -5.9047133707279436, -6.899575774863839, -7.9853292751836715, -8.760402281768812]

ERR = [1- (Drl[_]-GSE[_])/(D2[_]-GSE[_]) for _ in range(7)]
print(ERR)
IM = [Drl[_]/(D2[_]) for _ in range(7)]
print(IM)


P2 = [0.9968884120056379, 0.964694125248465, 0.9045072604175055, 0.8395075974780236, 0.6654283770546109, 0.5602042547812275, 0.39748611882738016]
P_rdm = [0.9999588869933728, 0.9922544234665865, 0.9168077974696554, 0.8972186719878226, 0.7353765629189636, 0.6468474646047149, 0.5806410381400212]
Prl = [0.9999988649542001, 0.998953842589463, 0.9961491624705469, 0.9733609214161482, 0.9146177800871382, 0.8805929286423865, 0.7193431861170122]

P2 = [np.sqrt(_) for _ in P2]
P_rdm = [np.sqrt(_) for _ in P_rdm]
Prl = [np.sqrt(_) for _ in Prl]


plt.figure(figsize=(6.4, 10.6))
plt.subplot(2,1,1)
p00, = plt.plot(N_list, D2, "d", color = colors[0], markersize = 9, linewidth=1.5)
p10, = plt.plot(N_list, D_rdm, "d", color = colors[4], markersize = 9, linewidth=1.5)
p20, = plt.plot(N_list, Drl, "d", color = colors[6], markersize = 9, linewidth=1.5)
p3 = plt.plot(N_list, GSE, "-.", color = "black", markersize = 4, linewidth=1.5)
# p4 = plt.plot(beta, Exp, "-", color = "black", markersize = 4, linewidth=1.5)


plt.legend([(p00), (p10),(p20)], ['QITE','Randomized QITE','RL-steered QITE'], numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)}, frameon=True, loc = 'best', fontsize = "x-large")

plt.title(r"(a)", loc="left", fontsize=18)
plt.xlabel(r"$N$", fontsize = "xx-large", fontname = "Times New Roman")
plt.ylabel(r"$E$", fontsize = "xx-large", fontname = "Times New Roman")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.ylim(-4.8 , -3.8)
# plt.xlim(-0.04, 0.94)



Ratio = [1.006252201301872, 1.0404395318358823, 1.1008207605783282, 1.1152614559114455, 1.1995884414130544, 1.2581165105210919, 1.2708695328774786]
plt.axes([0.23,0.576,0.25,0.1])
plt.plot(N_list, Ratio, color = colors[10], linewidth=1.5)
plt.xlabel("$N$",fontsize = "x-large")
plt.ylabel(r"$E_{RL}/E_{tsd}$",fontsize = "x-large")
plt.xticks([2,4,6,8],[2,4,6,8])
plt.yticks([1,1.1,1.2,1.3],[1,1.1,1.2,1.3])
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.subplot(2,1,2)
p00, = plt.plot(N_list, P2, "d", color = colors[0], markersize = 9, linewidth=1.5)
p10, = plt.plot(N_list, P_rdm, "d", color = colors[4], markersize = 9, linewidth=1.5)
p20, = plt.plot(N_list, Prl, "d", color = colors[6], markersize = 9, linewidth=1.5)
# p3 = plt.plot(beta, GSE, "-.", color = "black", markersize = 4, linewidth=1.5)
plt.legend([(p00), (p10),(p20)], ['QITE','Randomized QITE','RL-steered QITE'], numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)}, frameon=True, loc = 'best', fontsize = "x-large")


plt.title("(b)", loc='left', fontsize=18)
plt.xlabel(r"$N$", fontsize = "xx-large", fontname = "Times New Roman")
plt.ylabel(r"$F$", fontsize = "xx-large", fontname = "Times New Roman")

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.ylim(0.6 , 1)
# plt.xlim(-0.04, 0.94)


# plt.show()
plt.savefig("Fig5.pdf")
plt.close()

