import sys
sys.path.append('../Refactor/')
sys.path.append('../rbm-pytorch-refactor')

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import seaborn as sns
import torch
import argparse
import rbm_interface
import rbm_pytorch
import ising_methods_new
from torch.utils.data import DataLoader
from ising_methods_new import *
import json
from scipy.stats import norm

font = {'family' : 'normal',
        'weight' : 'light',
        'size'   : 10}

matplotlib.rc('font', **font)

sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()
sns.set_style('white')
sns.set_style('ticks',{"axes.linewidth": ".5", "xtick.minor.size" : ".5", "ytick.minor.size" : ".5","xtick.major.size" : "3", "ytick.major.size" : "3"})
sns.set_context('paper')

parse = argparse.ArgumentParser(description='Process some integers.')
parse.add_argument('--json', dest='input_json', default='params.json', help='JSON file describing the sample parameters',
					type=str)
parse.add_argument('--cuda', dest='cuda', type=bool, default=False)

args = parse.parse_args()

# Enable cuda
if args.cuda:
	dtype = torch.cuda.FloatTensor
else:
	dtype = torch.FloatTensor



L = 8

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
rbm10 = rbm_pytorch.RBM(n_vis=64, n_hid=parameters['n_hid'])
rbm10.load_state_dict(torch.load("batchsizes/10/trained_rbm.pytorch.200", map_location=lambda storage, loc: storage))
rbm100 = rbm_pytorch.RBM(n_vis=64, n_hid=parameters['n_hid'])
rbm100.load_state_dict(torch.load("batchsizes/100/trained_rbm.pytorch.200", map_location=lambda storage, loc: storage))
rbm1000 = rbm_pytorch.RBM(n_vis=64, n_hid=parameters['n_hid'])
rbm1000.load_state_dict(torch.load("batchsizes/1000/trained_rbm.pytorch.200", map_location=lambda storage, loc: storage))

states10 = ising_methods_new.sample_from_rbm(rbm10, parameters, dtype)
states100 = ising_methods_new.sample_from_rbm(rbm100, parameters, dtype)
states1000 = ising_methods_new.sample_from_rbm(rbm1000, parameters, dtype)

spin_states10 = convert_to_spins(states10)
spin_states100 = convert_to_spins(states100)
spin_states1000 = convert_to_spins(states1000)

mag_history10 = magnetisation(spin_states10).cpu().numpy()
energy_history10 = ising_energy(spin_states10, L).cpu().numpy()
mag_history100 = magnetisation(spin_states100).cpu().numpy()
energy_history100 = ising_energy(spin_states100, L).cpu().numpy()
mag_history1000 = magnetisation(spin_states1000).cpu().numpy()
energy_history1000 = ising_energy(spin_states1000, L).cpu().numpy()

split_mag10 = np.reshape(mag_history10, (-1,parameters['concurrent_states']))
split_energy10 = np.reshape(energy_history10, (-1,parameters['concurrent_states']))
split_mag100 = np.reshape(mag_history100, (-1,parameters['concurrent_states']))
split_energy100 = np.reshape(energy_history100, (-1,parameters['concurrent_states']))
split_mag1000 = np.reshape(mag_history1000, (-1,parameters['concurrent_states']))
split_energy1000 = np.reshape(energy_history1000, (-1,parameters['concurrent_states']))

susc10 = np.var(split_mag10, axis=1)/(N_spins * temperature)
susc100 = np.var(split_mag100, axis=1)/(N_spins * temperature)
susc1000 = np.var(split_mag1000, axis=1)/(N_spins * temperature)

sns.distplot(susc10, fit=norm, kde=False)
sns.distplot(susc100, fit=norm, kde=False)
sns.distplot(susc1000, fit=norm, kde=False)

plt.show()