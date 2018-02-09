import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

def right(i, L):
	"""Returns the index of the spin to the right of the spin at index i.
	Implements periodic boundary conditions.
	Args:
		i: Index of spin.
		L: Ising lattice size in an L x L lattice
	Returns:
		r: Index of spin to the right of spin i.
	"""
	j = i + 1
	return i + 1 if not j % L == 0 else i - (L-1)

def below(i, L):
	"""Returns the index of the spin below the spin at index i.
	Implements periodic boundary conditions.
	Args:
		i: Index of spin.
		L: Ising lattice size in an L x L lattice
	Returns:
		r: Index of spin below spin i.
	""" 
	j = i + 1
	return i + L if not j > (L**2 - L) else i - (L**2 - L) 

def ising_energy(spin_state, L):
	"""Returns the energy of an Ising configuration of spins.
	Args:
		spin_state: Torch tensor containing spin configurations in the (-1, 1) range
	Returns:
		e: Tensor of energies in the set of configurations
	"""
	# Cuda compatibility
	dtype = spin_state.type()

	# shape[0] is the number of concurrent states, shape[1] number of visible nodes
	e = torch.FloatTensor(spin_state.shape[0]).type(dtype)
	
	for i in range(spin_state.shape[1]):
		e.add_(-torch.mul(spin_state[:,i],(torch.add(spin_state[:,right(i,L)],spin_state[:,below(i,L)]))))

	return e

def convert_to_spins(state):
	"""Converts the configuration from (0,1) to (-1,1)
	Args:
		state: Tensor of states
	Returns:
		spin_state: Tensor of state in the (-1, 1) range.
	"""
	# Cuda compatibility
	dtype = state.type()

	return (2*state - torch.ones(state.shape).type(dtype))

def magnetisation(spin_state):
	"""Returns the absolute magnetisations of the states passed
	Args:
		spin_state: Torch tensor containing spin configurations in the (-1, 1) range
	Returns:
		Torch tensor containing the magnetisations of the states
	"""
	
	return torch.mean(spin_state, dim=1).abs_()

def ising_observables(states, L, temperature):
	"""Computes observables 
	"""
	states = convert_to_spins(states)
	mag_history = magnetisation(states).numpy()
	energy_history = ising_energy(states, L).numpy()


	N_spins = L**2
	avg_magnetisation = np.mean(mag_history)
	avg_energy = np.mean(energy_history)/N_spins
	susc = np.var(mag_history*N_spins) / (N_spins * temperature)
	heatc = np.var(energy_history) / (N_spins * temperature**2)
	return avg_magnetisation, susc, avg_energy, heatc

def sample_from_rbm(rbm, parameters, dtype=torch.FloatTensor, v_in=None):

	n_vis = parameters['ising']['size']**2
	states = torch.zeros(parameters['concurrent_states'],n_vis).type(dtype)

	# Initialise the gibbs chain with zeros or some input
	if v_in is not None:
		v = v_in
	else:
		v = torch.zeros(parameters['concurrent_states'], rbm.v_bias.data.shape[0]).type(dtype)

	for _ in range(parameters['thermalisation']):

		h, h_prob = hidden_from_visible(v, rbm.W.data, rbm.h_bias.data)
		v, v_prob = visible_from_hidden(h, rbm.W.data, rbm.v_bias.data)

	states.add_(v)

	for s in range(parameters['steps']):

		if (s % parameters['save interval'] == 0):

			save_images(parameters['image_dir'], s, v, v_prob, parameters['ising']['size'])

		for _ in range(parameters['autocorrelation']):
			h, h_prob = hidden_from_visible(v, rbm.W.data, rbm.h_bias.data)
			v, v_prob = visible_from_hidden(h, rbm.W.data, rbm.v_bias.data)

		states = torch.cat((states,v), dim=0)

	return states	

def sample_probability(prob, random):
	"""Get samples from a tensor of probabilities.

		:param probs: tensor of probabilities
		:param rand: tensor (of the same shape as probs) of random values
		:return: binary sample of probabilities
	"""
	torchReLu = nn.ReLU()
	return torchReLu(torch.sign(prob - random)).data


def hidden_from_visible(visible, W, h_bias):
	# Enable or disable neurons depending on probabilities
	probability = torch.sigmoid(F.linear(visible, W, h_bias))
	random_field = torch.rand(probability.size())
	new_states = sample_probability(probability, random_field)
	return new_states, probability


def visible_from_hidden(hid, W, v_bias):
	# Enable or disable neurons depending on probabilities
	probability = torch.sigmoid(F.linear(hid, W.t(), v_bias))
	random_field = torch.rand(probability.size())
	new_states = sample_probability(probability, random_field)
	return new_states, probability		

def save_images(directory, step, v, v_prob, L, output_states=True):

	if output_states:
		imgshow(directory + "dream" + str(step).zfill(6),
				make_grid(v.view(-1, 1, L, L)))
	else:
		imgshow(directory + "dream" + str(step).zfill(6),
				make_grid(v_prob.view(-1, 1, L, L)))

def imgshow(file_name, img):
	npimg = np.transpose(img.numpy(), (1, 2, 0))
	f = "./%s.png" % file_name
	Wmin = img.min
	Wmax = img.max
	plt.imsave(f, npimg, vmin=Wmin, vmax=Wmax)