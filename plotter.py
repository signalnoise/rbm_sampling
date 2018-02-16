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

parser = argparse.ArgumentParser(description="Process data files")
parser.add_argument('--loss_timeline', dest='loss', type=str)
parser.add_argument('--training_data', dest='train', type=str)
parser.add_argument('--thermo_data', dest='thermo', type=str)
args = parser.parse_args()

with open(args.train) as file:
	temperature = float(file.readline())

train_loader = DataLoader(rbm_interface.CSV_Ising_dataset(args.train, size=8), 
							shuffle=True, batch_size=100000, drop_last=True)
for i, (data, target) in enumerate(train_loader):
	mag, susc, energy, heatc = ising_methods_new.ising_observables(data, 8, temperature)

#names = ['Epochs', 'Magnetisation', 'Susceptibility', 'Energy', 'Heat Capacity']
#loss_names = ['Epochs', 'Loss', 'Free Energy', 'Reconstruction Error']
#thermo_df = pd.read_csv(args.thermo, sep="\t", names=names)
#loss_df = pd.read_csv(args.loss, sep="\t", skiprows=1,names=loss_names)

with open("convtempdata.txt", "a") as file:
	file.write("{:f}\t".format(temperature))
	for i in range(1, 5, 1):
		file.write("{:f}\t".format(thermo_df.iloc[100].as_matrix()[i]))
	file.write("\n")
# plt.fill_between(thermo_df.Epochs, thermo_df.Magnetisation - thermo_df.Susceptibility, thermo_df.Magnetisation + thermo_df.Susceptibility, alpha=0.2)

# plt.plot(loss_df.Epochs, loss_df['Reconstruction Error'])
"""
plt.tight_layout()
plt.show()
"""