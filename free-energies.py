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
sns.set_style('ticks', {"axes.linewidth": ".5", "xtick.minor.size" : ".5", "ytick.minor.size" : ".5","xtick.major.size" : "-5", "ytick.major.size" : "-5"})

path = "data/adadelta/poorly-trained/"
path2 = "data/sgd/well-trained/"

names = ["Epochs", "Loss", "Mean Free Energy", "Reconstruction Error",
		 "Validation Free Energy", "Comparison Free Energy"]

df1 = pd.read_csv(path + "Loss_timeline.data", sep="\t", names=names, skiprows=1)

df2 = pd.read_csv(path2 + "Loss_timeline_lr_0.01.data", sep="\t", names=names, skiprows=1)

plt.plot(df1[names[0]], df1[names[4]]/df1[names[5]])
#plt.plot(df1[names[0]], df1[names[5]])
plt.plot(df2[names[0]], df2[names[4]]/df2[names[5]])
#plt.plot(df2[names[0]], df2[names[5]])
plt.show()