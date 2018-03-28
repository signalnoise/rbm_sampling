from ising_methods_new import *
import json
import argparse
import sys
sys.path.append('../Refactor/')
import rbm_pytorch
import torch
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--json', dest='input_json', default='params.json', help='JSON file describing the sample parameters',
					type=str)

args = parser.parse_args()

try:
	parameters = json.load(open(args.input_json))
except IOError as e:
	print("I/O error({0}) (json): {1}".format(e.errno, e.strerror))
	sys.exit(1)
except:
	print("Unexpected error:", sys.exc_info()[0])
	raise
L = 8
temperature = 2.27
N_bootstrap = 10000
dtype = torch.FloatTensor

rbm = rbm_pytorch.RBM(n_vis=parameters['ising']['size']**2, n_hid=parameters['n_hid'])
rbm.load_state_dict(torch.load(parameters['saved_state'], map_location=lambda storage, loc: storage))

states = sample_from_rbm(rbm,parameters)
spin_states = convert_to_spins(states)
mag_history = magnetisation(spin_states).cpu().numpy()
energy_history = ising_energy(spin_states, L).cpu().numpy()

N_spins = spin_states.shape[2]
avg_magnetisation = average_magnetisation(mag_history, N_spins)
avg_energy = average_energy(energy_history, N_spins)
susc = susceptibility(mag_history, N_spins, temperature)
heatc = heat_capacity(energy_history, N_spins, temperature)



#print(mag_history.shape)

(m,merr), (ch, cherr), (e, eerr), (hc, hcerr) = ising_observables(states, 8, 2.27)

x = [x for x in range(64)]
plt.errorbar(x=x, fmt='o', y=susc)
print(np.std(susc))
plt.axhline(ch+cherr, color='k', ls='--')
plt.fill_between(x,ch-(cherr/np.sqrt(64)), ch+(cherr/np.sqrt(64)), alpha=0.4)
plt.axhline(ch, color='k')
plt.axhline(ch-cherr, color='k', ls='--')

plt.title('Susceptibility')
plt.xlabel('Machines')
plt.ylabel('$\chi$')
plt.show()
