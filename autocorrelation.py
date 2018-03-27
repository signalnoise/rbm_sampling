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
import rbm_pytorch
import ising_methods_new
from torch.utils.data import DataLoader
from ising_methods_new import *
import json
#from pandas.plotting import autocorrelation_plot


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

def autocorrelation_plot(series, lw=1, ax=None, **kwds):
    """Autocorrelation plot for time series.
    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    kwds : keywords
        Options to pass to matplotlib plotting method
    Returns:
    -----------
    ax: Matplotlib axis object
    """
    import matplotlib.pyplot as plt
    from pandas.compat import lmap
    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca(xlim=(1, n), ylim=(-1.0, 1.0))
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) *
                (data[h:] - mean)).sum() / float(n) / c0
    x = np.arange(n) + 1
    y = lmap(r, x)
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle='--', color='grey', linewidth=lw)
    ax.axhline(y=z95 / np.sqrt(n), color='grey', linewidth=lw)
    ax.axhline(y=0.0, color='black', linewidth=lw)
    ax.axhline(y=-z95 / np.sqrt(n), color='grey', linewidth=lw)
    ax.axhline(y=-z99 / np.sqrt(n), linestyle='--', color='grey', linewidth=lw)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    return ax


font = {'family' : 'normal',
        'weight' : 'light',
        'size'   : 10}

matplotlib.rc('font', **font)

sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()
sns.set_style('ticks', {"axes.linewidth": ".5", "xtick.minor.size" : ".5", "ytick.minor.size" : ".5","xtick.major.size" : "2", "ytick.major.size" : "2"
	,"xtick.direction" : "in", "ytick.direction" : "in"})
sns.set_context("notebook")


parse = argparse.ArgumentParser(description='Process some integers.')
parse.add_argument('--json', dest='input_json', default='params.json', help='JSON file describing the sample parameters',
					type=str)

args = parse.parse_args()

ymin = -0.4

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
rbm.load_state_dict(torch.load(parameters['saved_state'], map_location=lambda storage, loc: storage))

states = ising_methods_new.sample_from_rbm(rbm, parameters, dtype)

spin_states = convert_to_spins(states)

mag_history = magnetisation(spin_states).cpu().numpy()
energy_history = ising_energy(spin_states, L).cpu().numpy()

split_mag = np.reshape(mag_history, (-1,parameters['concurrent_states']))
split_energy = np.reshape(energy_history, (-1,parameters['concurrent_states']))

split_susc = np.var(split_mag, axis=1)/(N_spins * temperature)
split_heatc = np.var(split_energy,axis=1)/(N_spins * temperature**2)

split_mag = np.mean(split_mag, axis=1)
#split_mag = pd.Series(split_mag)

split_energy = np.mean(split_energy, axis=1)
#split_energy = pd.Series(split_energy)

names = ["Magnetisation", "Energy", "Susceptibility", "Heat Capacity"]

with open("autocorrelationw_data_" + str(parameters['thermalisation']) + ".txt", "w") as file:
	file.write("\t".join(names) + "\n")
	for i in range(len(split_mag)):
		text = "{:f}\t{:f}\t{:f}\t{:f}\n".format(split_mag[i], split_energy[i], split_susc[i], split_heatc[i])
		file.write(text)

	


"""
autocorrelation_plot(split_mag)
plt.grid(b=False)
plt.title("Magnetisation")
plt.gca().set_ylim([ymin,1.0])
plt.gca().title.set_position([0.88,0.92])
plt.tight_layout()
plt.show()


autocorrelation_plot(split_energy)
plt.grid(b=False)
plt.title("Energy")
plt.gca().set_ylim([ymin,1.0])
plt.gca().title.set_position([0.88,0.92])
plt.tight_layout()
plt.show()

autocorrelation_plot(split_susc)
plt.grid(b=False)
plt.title("Susceptibility")
plt.gca().set_ylim([ymin,1.0])
plt.gca().title.set_position([0.88,0.92])
plt.tight_layout()
plt.show()

autocorrelation_plot(split_heatc)
plt.grid(b=False)
plt.title("Heat Capacity")
plt.gca().set_ylim([ymin,1.0])
plt.gca().title.set_position([0.88,0.92])
plt.tight_layout()
plt.show()
 
"""
"""
mag_correlation = []
energy_correlation = []
for i in range(250):
	mag_correlation.append(autocorrelation(split_mag,i))

print(correlation_time(split_mag))
plt.plot(range(250), mag_correlation)
plt.show()

"""