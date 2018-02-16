import sys
sys.path.append('../Refactor/')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import torch
import argparse
import rbm_interface
import ising_methods_new
from torch.utils.data import DataLoader

font = {'family' : 'normal',
        'weight' : 'light',
        'size'   : 10}

matplotlib.rc('font', **font)

sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()

names = ['Temperature', 'Magnetisation', 'Susceptibility', 'Energy', 'Heat Capacity']
df = pd.read_csv("tempdata.txt", sep="\t", names=names)
df2 = pd.read_csv("convtempdata.txt", sep="\t", names=names)

print(df)
print(df2)

plt.plot(df.Temperature, df['Energy'])
plt.plot(df2.Temperature, df2['Energy'])

plt.tight_layout()
plt.show()