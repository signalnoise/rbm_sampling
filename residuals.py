import sys
sys.path.append('../Refactor/')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import torch
import argparse
import rbm_interface
import ising_methods_new
from torch.utils.data import DataLoader



sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()
sns.set_style('white')
sns.set_style('ticks', {"axes.linewidth": ".5", "xtick.minor.size" : ".5", "ytick.minor.size" : ".5","xtick.major.size" : "2", "ytick.major.size" : "2",
						"xtick.direction" : "in", "ytick.direction" : "in"})

"""
font =  {'weight' : 'light',
        'size'   : 10}
matplotlib.rc('font', **font)
"""
sns.set_context("paper")

path = "adagrad/"

names = ['Temperature', 'Magnetisation', 'merr', 'Susceptibility', 'cherr', 'Energy', 'eerr', 'Heat Capacity', 'herr']
df = pd.read_csv(path + "temp_graph_machines.txt", sep="\t", names=names)
df2 = pd.read_csv(path + "training_data_observables.txt", sep="\t", names=names)

#print(df)

#quant = 5

fig, axes = plt.subplots(2, 2, sharex='all', figsize=(8,8))
gs1 = gridspec.GridSpec(2, 2)
gs1.update(hspace=0.004, wspace=0.22)
axes = axes.ravel()

for i in range(4):
	quant = 2*i + 1
	axes[i] = plt.subplot(gs1[i])
	axes[i].errorbar(df.Temperature, df[names[quant]]-df2[names[quant]], yerr=df[names[quant + 1]], fmt='o', ms='6', alpha=0.8)
	axes[i].axhline(0, ls='--', color=palette[0])
	axes[i].fill_between(df.Temperature,-df[names[quant + 1]], df[names[quant + 1]], color=palette[1], alpha=0.4)
	axes[i].fill_between(df.Temperature,-2*df[names[quant + 1]], 2*df[names[quant + 1]], color=palette[1], alpha=0.3)
	axes[i].linewidth=0.5
	axes[i].xaxis.set_ticks_position('both')
	axes[i].yaxis.set_ticks_position('both')
	#axes[i].grid(linewidth=0.5)
	#axes[i].set_xticklabels([])
	#axes[i].set_yticklabels([])
    #axes[i].set_xlabel(names[0])
	axes[i].set_ylabel(names[quant])
	


axes[2].set_xlabel(names[0])
axes[3].set_xlabel(names[0])

"""

plt.errorbar(df.Temperature, df[names[quant]], yerr=df[names[quant + 1]], fmt='o', ms='5')
plt.plot(df2.Temperature, df2[names[quant]], '--')
plt.xlabel(names[0])
plt.ylabel(names[quant])
"""
#plt.tight_layout()
sns.despine()
#plt.savefig("ob_v_t.png", dpi=300)
plt.show()