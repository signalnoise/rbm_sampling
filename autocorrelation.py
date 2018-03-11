import sys
sys.path.append('../Refactor/')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import torch
import argparse
import rbm_interface
import rbm_pytorch
import ising_methods_new
from torch.utils.data import DataLoader
from ising_methods_new import *
import json
from pandas.plotting import autocorrelation_plot


def autocorrelation(split_history, T):
	s = 0
	for k in range(len(split_history[:,0])-T):
		product = np.multiply(split_history[k,:],split_history[k+T,:])
		s = s + (np.mean(product) - np.power(np.mean(split_history[k,:]),2))/np.var(split_history[k,:])
	return s

def correlation_time(split_history):
	s = 0
	for T in range(250):
		s = s + autocorrelation(split_history, T)
	s = s/autocorrelation(split_history, 0)
	return s

font = {'family' : 'normal',
        'weight' : 'light',
        'size'   : 10}

matplotlib.rc('font', **font)

sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()
sns.set_style('white', {"axes.linewidth": ".5", "xtick.minor.size" : ".5", "ytick.minor.size" : ".5","xtick.major.size" : "-5", "ytick.major.size" : "-5"})
sns.set_context("paper")


parse = argparse.ArgumentParser(description='Process some integers.')
parse.add_argument('--json', dest='input_json', default='params.json', help='JSON file describing the sample parameters',
					type=str)

args = parse.parse_args()

L = 8
N_bootstrap = 10000
N_spins = 64
temperature = 2.27

try:
	parameters = json.load(open(args.input_json))
except IOError as e:
	print("I/O error({0}) (json): {1}".format(e.errno, e.strerror))
	sys.exit(1)
except:
	print("Unexpected error:", sys.exc_info()[0])
	raise

dtype = torch.FloatTensor
rbm = rbm_pytorch.RBM(n_vis=64, n_hid=parameters['n_hid'])
rbm.load_state_dict(torch.load(parameters['saved_state']))

states = ising_methods_new.sample_from_rbm(rbm, parameters, dtype)

spin_states = convert_to_spins(states)

mag_history = magnetisation(spin_states).cpu().numpy()
energy_history = ising_energy(spin_states, L).cpu().numpy()

split_mag = np.reshape(mag_history, (-1,parameters['concurrent_states']))
split_energy = np.reshape(energy_history, (-1,parameters['concurrent_states']))

split_susc = np.var(split_mag, axis=1)/(N_spins * temperature)
split_heatc = np.var(split_energy,axis=1)/(N_spins * temperature**2)

split_mag = np.mean(split_mag, axis=1)
split_mag = pd.Series(split_mag)
autocorrelation_plot(split_mag)
plt.grid(b=False)
plt.gca().set_ylim([-0.3,1.0])
plt.show()

split_energy = np.mean(split_energy, axis=1)
split_energy = pd.Series(split_energy)
autocorrelation_plot(split_energy)
plt.grid(b=False)
plt.gca().set_ylim([-0.3,1.0])
plt.show()

autocorrelation_plot(split_susc)
plt.grid(b=False)
plt.gca().set_ylim([-0.3,1.0])
plt.show()

autocorrelation_plot(split_heatc)
plt.grid(b=False)
plt.gca().set_ylim([-0.3,1.0])
plt.show()


"""
mag_correlation = []
energy_correlation = []
for i in range(250):
	mag_correlation.append(autocorrelation(split_mag,i))

print(correlation_time(split_mag))
plt.plot(range(250), mag_correlation)
plt.show()

"""