import sys
sys.path.append('../Refactor/')
sys.path.append('../rbm-pytorch-refactor')

import matplotlib
matplotlib.use('Agg')
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
rbm = rbm_pytorch.RBM(n_vis=64, n_hid=parameters['n_hid'])
rbm.load_state_dict(torch.load(parameters['saved_state'], map_location=lambda storage, loc: storage))

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

with open('data/therm/magnetisation_' + str(parameters['thermalisation']) + '.txt') as file:
	x = [str(item) for item in split_mag]
	file.write('\n'.join(x))


'''
susc = np.var(split_mag, axis=1)/(N_spins * temperature)
print(susc.shape)
#energy = torch.mean(energy_history)/N_spins
heatc = np.var(split_energy, axis=1)/(N_spins * temperature**2)
print(heatc.shape)
'''


'''
with open("data/therm/with-autocorr/adadheatcapacity_" + str(parameters['thermalisation']) + ".txt", 'w') as file:
	x = [str(item) for item in heatc]
	file.write("\n".join(x))

with open("data/therm/with-autocorr/adadsusceptibility_" + str(parameters['thermalisation']) + ".txt",'w') as file:
	y = [str(item) for item in susc]
	file.write("\n".join(y))
'''

"""
labels = ['Magneto Value', 'Gaussian Fit']

#print("sigma: {:f}".format(np.std(heatc.numpy())))
plt.ylabel('Frequency')
plt.axvline(0.943410,ls='--', color='k', linewidth=0.7)#(1.151843, ls='--', color='k', linewidth=0.7)
sns.distplot(heatc, fit=norm, kde=False, fit_kws={"color": palette[2]})
"""
"""
mu = np.mean(heatc)
sigma = np.sqrt(np.var(heatc))
x = np.linspace(np.amin(heatc), np.amax(heatc), 100)
plt.hist(heatc, 20)

plt.axvline(1.14904, ls='--', color=palette[1])
plt.plot(x, mlab.normpdf(x , mu, sigma))
"""
"""
plt.legend(labels)
plt.xlabel('Heat Capacity')
sns.despine()
plt.tight_layout()
plt.savefig("heatc-ada.png")

plt.clf()

#print("sigma: {:f}".format(np.std(susc.numpy())))
plt.ylabel('Frequency')
plt.axvline(0.696052,ls='--', color='k', linewidth=0.7)#axvline(1.225381, ls='--', color='k', linewidth=0.7)
sns.distplot(susc, fit=norm, kde=False,  fit_kws={"color": palette[2]})
"""
"""
mu = np.mean(susc)
sigma = np.sqrt(np.var(susc))
plt.hist(susc, 20)
plt.xlabel('Susceptibility')
"""
"""
plt.legend(labels)
plt.xlabel('Susceptibility')
sns.despine()
plt.tight_layout()
plt.savefig("susc-ada.png")
"""