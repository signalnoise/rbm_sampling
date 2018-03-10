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

font = {'family' : 'normal',
        'weight' : 'light',
        'size'   : 10}

matplotlib.rc('font', **font)

sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()

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
'''
mag_samples = bootstrap_sample(mag_history, N_bootstrap, dtype)
energy_samples = bootstrap_sample(energy_history, N_bootstrap, dtype)
'''
#mag = torch.mean(mag_history)/N_spins
susc = np.var(split_mag, axis=1)/(N_spins * temperature)
print(susc.shape)
#energy = torch.mean(energy_history)/N_spins
heatc = np.var(split_energy, axis=1)/(N_spins * temperature**2)
print(heatc.shape)

plt.ylabel('Frequency')
plt.axvline(0.776831, ls='--', color=palette[1])
#print("sigma: {:f}".format(np.std(mag.numpy())))
plt.hist(mag_history/N_spins, 15)
plt.xlabel('Magnetisation')
plt.tight_layout()
plt.show()
#print("sigma: {:f}".format(np.std(energy.numpy())))
plt.ylabel('Frequency')
plt.hist(energy_history/N_spins, 15)
plt.axvline(-1.485561, ls='--', color=palette[1])
plt.xlabel('Energy')
plt.tight_layout()
plt.show()


#print("sigma: {:f}".format(np.std(heatc.numpy())))
plt.ylabel('Frequency')
plt.hist(heatc, 20)
plt.axvline(1.14904, ls='--', color=palette[1])
plt.xlabel('Heat Capacity')
plt.tight_layout()
plt.show()

#print("sigma: {:f}".format(np.std(susc.numpy())))
plt.ylabel('Frequency')
plt.hist(susc, 20)
plt.xlabel('Susceptibility')
plt.axvline(1.20866, ls='--', color=palette[1])
plt.tight_layout()
plt.show()
