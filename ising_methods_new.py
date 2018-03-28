import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import math

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

	# shape[0] is the number of concurrent states, shape[1] the number of steps,shape[2] number of visible nodes
	e = torch.zeros(spin_state.shape[0],spin_state.shape[1]).type(dtype)
	
	for i in range(spin_state.shape[2]):
		e = torch.add(e,-torch.mul(spin_state[:,:,i],(torch.add(spin_state[:,:,right(i,L)],spin_state[:,:,below(i,L)]))))
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
	
	return torch.mean(spin_state, dim=2).abs_()*spin_state.shape[2]

def ising_observables(states, L, temperature):
	"""Computes observables given a full dataset for a certain temperature.
	Args:
		states: Tensor containing a full set of configurations in the (0,1) range
		L: L: Ising lattice size in an L x L lattice
		temperature: The temperature at which the observables should be computed.
	Returns:
		avg_magnetisation: The average scaled magnetisation for the data set.
		susc: The susceptibility of the data set.
		avg_energy: The average scaled energy for the data set.
		heatc: The heat capacity of the data set.
	"""
	spin_states = convert_to_spins(states)
	dtype = states.type()

	# Convert the torch tensor to numpy to run the statistics
	mag_history = magnetisation(spin_states).cpu().numpy()
	energy_history = ising_energy(spin_states, L).cpu().numpy()

	N_spins = spin_states.shape[2]
	avg_magnetisation = average_magnetisation(mag_history, N_spins)
	avg_energy = average_energy(energy_history, N_spins)
	susc = susceptibility(mag_history, N_spins, temperature)
	heatc = heat_capacity(energy_history, N_spins, temperature)

	mag_err, susc_err, energy_err, heatc_err = ising_errors(avg_magnetisation, avg_energy, susc, heatc, 3000, dtype)
	
	mag_t = avg_magnetisation.mean(), mag_err
	susc_t = susc.mean(), susc_err
	energy_t = avg_energy.mean(), energy_err
	heatc_t = heatc.mean(), heatc_err

	return mag_t, susc_t, energy_t, heatc_t

def average_magnetisation(mag_history, N_spins):
	return np.mean(mag_history, axis=1, dtype=np.float64)/N_spins

def average_energy(energy_history, N_spins):
	return  np.mean(energy_history, axis=1, dtype=np.float64)/N_spins

def susceptibility(mag_history, N_spins, temperature):
	return np.var(mag_history, axis=1, dtype=np.float64) / (N_spins * temperature)

def heat_capacity(energy_history, N_spins, temperature):
	return np.var(energy_history, axis=1, dtype=np.float64) / (N_spins * temperature**2)

def ising_errors(mag_history, energy_history, susc_history, heatc_history, N_bootstrap, dtype):
	
	mag_samples = bootstrap_sample(mag_history, N_bootstrap, dtype)
	energy_samples = bootstrap_sample(energy_history, N_bootstrap, dtype)
	susc_samples = bootstrap_sample(susc_history, N_bootstrap, dtype)
	heatc_samples = bootstrap_sample(heatc_history, N_bootstrap, dtype)


	mag = torch.mean(mag_samples, dim=1)
	susc = torch.mean(susc_samples, dim=1)
	energy = torch.mean(energy_samples, dim=1)
	heatc = torch.mean(heatc_samples, dim=1)

	mag_err = np.std(mag.numpy())
	susc_err = np.std(susc.numpy())
	energy_err = np.std(energy.numpy())
	heatc_err = np.std(heatc.numpy())

	return mag_err, susc_err, energy_err, heatc_err

def bootstrap_sample(nd_array, N_bootstrap, dtype):
	indices = np.random.randint(len(nd_array), size = (N_bootstrap, len(nd_array)))
	output = torch.from_numpy(np.take(nd_array,indices)).type(dtype)
	return output


def sample_from_rbm(rbm, parameters, dtype=torch.FloatTensor, v_in=None, save_images=False):
	""" Draws samples from an rbm.
	Args:
		rbm: a trained instance of rbm_pytorch.rbm
		parameters: Json parameters
		dtype: cuda or cpu torch tensor.
		v_in: seed for the gibbs chain.
	Returns:
		states: a tensor containing all samples of shape (n_samples, n_visible)
	"""
	
	n_vis = parameters['ising']['size']**2

	# Create a zeroed tensor 
	states = torch.zeros(parameters['concurrent_states'],1,n_vis).type(dtype)

	# Initialise the gibbs chain with zeros or some input
	if v_in is not None:
		v = v_in.type(dtype)
		for i in range(parameters['concurrent_states']-1):
			v = torch.cat((v,v_in.type(dtype)), dim=0)
	else:
		#v = torch.zeros(parameters['concurrent_states'], rbm.v_bias.data.shape[0]).type(dtype)
		v = F.relu(torch.sign(torch.rand(parameters['concurrent_states'], 1, n_vis)-0.5)).data.type(dtype)

	# Run the gibbs chain for a certain number of steps to allow it to converge to the
	# stationary distribution.
	for _ in range(parameters['thermalisation']):

		v, v_prob = new_state(v, rbm, dtype)

	# Fill the empty tensor with the first sample from the machine
	states.add_(v)

	# Draw a number of samples equal to the number steps
	for s in range(parameters['steps']):

		if (s % parameters['save interval'] == 0) and (save_images):

			save_imgs(parameters['image_dir'], s, v, v_prob, parameters['ising']['size'])

		# Run the gibbs chain for a given number of steps to reduce correlation between samples
		for _ in range(parameters['autocorrelation']):
			v, v_prob = new_state(v, rbm, dtype)

		v, v_prob = new_state(v, rbm, dtype)

		# Concatenate the samples
		states = torch.cat((states,v), dim=1)

	return states	

def sample_probability(probabilities, random):
	"""Get samples from a tensor of probabilities.
	Args:
		probs: tensor of probabilities
		rand: tensor (of the same shape as probs) of random values
	Returns:
		binary sample of probabilities
	"""
	torchReLu = nn.ReLU()
	return torchReLu(torch.sign(probabilities - random)).data


def hidden_from_visible(visible, W, h_bias, dtype):
	"""Samples the hidden (latent) variables given the visible.

    Args:
		visible: Tensor containing the states of the visible nodes
		W: Weights of the rbm.
		h_bias: Biases of the hidden nodes of the rbm.
	
    Returns:
		new_states: Tensor containing binary (1 or 0) states of the hidden variables
		probability: Tensor containing probabilities P(H_i = 1| {V})
	"""
	probability = torch.sigmoid(F.linear(visible, W, h_bias))
	random_field = torch.rand(probability.size()).type(dtype)
	new_states = sample_probability(probability, random_field)
	return new_states, probability


def visible_from_hidden(hid, W, v_bias, dtype):
	"""Samples the hidden (latent) variables given the visible.

	Args:
		hid: Tensor containing the states of the hidden nodes
		W: Weights of the rbm.
		v_bias: Biases of the visible nodes of the rbm.
	
	Returns:
		new_states: Tensor containing binary (1 or 0) states of the visible variables
		probability: Tensor containing probabilities P(V_i = 1| {H})
	"""
	probability = torch.sigmoid(F.linear(hid, W.t(), v_bias))
	random_field = torch.rand(probability.size()).type(dtype)
	new_states = sample_probability(probability, random_field)
	return new_states, probability

def new_state(v, rbm, dtype):
	"""
	"""	
	h, h_prob = hidden_from_visible(v, rbm.W.data.type(dtype), rbm.h_bias.data.type(dtype), dtype)
	v, v_prob = visible_from_hidden(h, rbm.W.data.type(dtype), rbm.v_bias.data.type(dtype), dtype)
	return v, v_prob	

def save_imgs(directory, step, v, v_prob, L, output_states=True):
	""" Saves images of the generated states.
	Args:
		directory: Directory in which to save the images.
		step: The current step in the gibbs chain.
		v: Tensor containing states of visible nodes.
		v_prob: Tensor containing probabilities P(V_i = 1| {H})
		L: Ising lattice size in an L x L lattice.
		output_states: Boolean determining whether to output states or probabilities.
	"""
	if output_states:
		imgshow(directory + "dream" + str(step).zfill(6),
				make_grid(v.cpu().view(-1, 1, L, L)))
	else:
		imgshow(directory + "dream" + str(step).zfill(6),
				make_grid(v_prob.cpu().view(-1, 1, L, L)))

def imgshow(file_name, img):
	""" Saves images.
	"""
	npimg = np.transpose(img.numpy(), (1, 2, 0))
	f = "./%s.png" % file_name
	Wmin = img.min
	Wmax = img.max
	plt.imsave(f, npimg, vmin=Wmin, vmax=Wmax)