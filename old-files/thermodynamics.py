from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

from tqdm import *

import argparse
import torch
import torch.utils.data
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from math import exp, sqrt

import json
from pprint import pprint

import rbm_pytorch

from ising_methods import heat_capacity, IsingState

def get_ising_variables(field, sign=-1):
	""" Get the Ising variables {-1,1} representation
	of the RBM Markov fields

	:param field: the RBM state (visible or hidden), numpy

	:param sign: sign of the conversion

	:return: the Ising field

	"""
	sign_field = np.full(field.shape, sign)

	return (2.0 * field + sign_field).astype(int)


def ising_magnetization(field):
	m = np.abs((field).mean())
	return np.array([m, m * m])


def ising_averages(mag_history, energy_history, model_size, temp, label=""):
	sample_size = mag_history.size
	mag_avg = mag_history.mean(axis=0, keepdims=True)  # average of m and m^2
	mag_std = mag_history.std(axis=0, keepdims=True)  # std error of m and m^2

	susc = np.var(mag_history[:, 0] * model_size) / (model_size * temp)
	heatc = heat_capacity(energy_history, temp, model_size)
	return mag_avg[0, 0], susc, energy_history.mean(), heatc


def imgshow(file_name, img):
	npimg = np.transpose(img.numpy(), (1, 2, 0))
	f = "./%s.png" % file_name
	Wmin = img.min
	Wmax = img.max
	plt.imsave(f, npimg, vmin=Wmin, vmax=Wmax)


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


def sample_from_rbm(steps, model, image_size, nstates=30, v_in=None, savename=None):
	""" Samples from the RBM distribution function

		:param steps: Number of Gibbs sampling steps.
		:type steps: int

		:param model: Trained RBM model.
		:type model: RBM class

		:param image_size: Linear size of output images
		:type image_size: int

		:param nstates: Number of states to generate concurrently
		:type nstates: int

		:param v_in: Initial states (optional)

		:return: Last generated visible state
	"""

	if (parameters['initialize_with_training']):
		v = v_in
	else:
		# Initialize with zeroes
		# v = torch.zeros(nstates, model.v_bias.data.shape[0])
		# Random initial visible state
		v = F.relu(torch.sign(torch.rand(nstates, model.v_bias.data.shape[0]) - 0.5)).data

	v_prob = v

	magv = []
	magh = []
	energy = []
	# Run the Gibbs sampling for a number of steps
	for s in range(steps + parameters['thermalization']):
		# r = np.random.random()
		# if (r > 0.5):
		#    vin = torch.zeros(nstates, model.v_bias.data.shape[0])
		#    vin = torch.ones(nstates, model.v_bias.data.shape[0])
		# else:


		if (s % parameters['save interval'] == 0):
			if savename is not None:
				imgshow(parameters['image_dir'] + savename + str(s).zfill(6),
				        make_grid(v.view(-1, 1, image_size, image_size)))
			elif parameters['output_states']:
				imgshow(parameters['image_dir'] + "dream" + str(s).zfill(6),
						make_grid(v.view(-1, 1, image_size, image_size)))
			else:
				imgshow(parameters['image_dir'] + "dream" + str(s).zfill(6),
						make_grid(v_prob.view(-1, 1, image_size, image_size)))
			if args.verbose:
				print(s, "OK")

		# Update k steps
		h, h_prob = hidden_from_visible(v, model.W.data, model.h_bias.data)
		v, v_prob = visible_from_hidden(h, model.W.data, model.v_bias.data)
		# vin = v


		# Save data
		if (s > parameters['thermalization']):
			for j in range(v.shape[0]):
				magv.append(ising_magnetization(get_ising_variables(v[j, :].numpy())))
				magh.append(ising_magnetization(get_ising_variables(h[j, :].numpy())))
				energy.append(IsingState(v[j,:],8).energy())
			for _ in range(parameters['autocorrelation']):
				h, h_prob = hidden_from_visible(v, model.W.data, model.h_bias.data)
				v, v_prob = visible_from_hidden(h, model.W.data, model.v_bias.data)

	return v, np.asarray(magv), np.asarray(magh), np.asarray(energy)


# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--json', dest='input_json', default='params.json', help='JSON file describing the sample parameters',
					type=str)
parser.add_argument('--verbose', dest='verbose', default=False, help='Verbosity control',
					type=bool, choices=[False, True])

args = parser.parse_args()
try:
	parameters = json.load(open(args.input_json))
except IOError as e:
	print("I/O error({0}) (json): {1}".format(e.errno, e.strerror))
	sys.exit(1)
except:
	print("Unexpected error:", sys.exc_info()[0])
	raise

if args.verbose:
	print(args)
	pprint(parameters)

n_h = parameters['hidden_size'] * parameters['hidden_size']


# For the MNIST data set
if parameters['model'] == 'mnist':
	model_size = MNIST_SIZE
	image_size = 28
	if parameters['initialize_with_training']:
		print("Loading MNIST training set...")
		train_loader = datasets.MNIST('./DATA/MNIST_data', train=True, download=True,
									  transform=transforms.Compose([
										  transforms.ToTensor()
									  ]))
#############################
elif parameters['model'] == 'ising':
	# For the Ising Model data set
	image_size = parameters['ising']['size']
	model_size = image_size * image_size
	if parameters['initialize_with_training']:
		print("Loading Ising training set...")
		train_loader = rbm_pytorch.CSV_Ising_dataset(parameters['ising']['train_data'], size=model_size)

		# Compute magnetization and susceptibility
		train_mag = []
		train_energies = []
		for i in range(len(train_loader)):
			data = train_loader[i][0].view(-1, model_size)
			train_mag.append(ising_magnetization(get_ising_variables(data.numpy())))
			train_energies.append(IsingState(data,8).energy())
		tr_magarray= np.asarray(train_mag)
		tr_energies = np.asarray(train_energies)
		mag , susc, avg_e, heatc = ising_averages(tr_magarray, tr_energies, model_size, parameters['temperature'], "training_set")
		print("Set Magnetisation: {:f}, \t Set Susceptibility: {:f}, \t Set Heat Capacity: {:f}".format(mag, susc, heatc))

# Read the model, example
rbm = rbm_pytorch.RBM(n_vis=model_size, n_hid=n_h)

for x in range(10, 60, 10):

	try:
		state = parameters['state_dir']+"trained_rbm.pytorch." + str(x)
		print("Loading " + state + "...")
		rbm.load_state_dict(torch.load(state))
	except IOError as e:
		print("I/O error({0}) (states): {1}".format(e.errno, e.strerror))
		sys.exit(1)
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

	print('Model succesfully loaded')

	if parameters['initialize_with_training']:
		data = torch.zeros(parameters['concurrent samples'], model_size)
		for i in range(parameters['concurrent samples']):
			data[i] = train_loader[i + 100][0].view(-1, model_size)
		v, magv, magh, energies = sample_from_rbm(parameters['steps'], rbm, image_size, v_in=data, savename=str(x).zfill(5))
	else:
		v, magv, magh, energies = sample_from_rbm(
			parameters['steps'], rbm, image_size, parameters['concurrent samples'], savename=str(x).zfill(5))

	mag, susc, avg_e, heatc = ising_averages(magv, energies, model_size, parameters['temp'], "v")


	with open("./sgd_mixed7.txt", "a") as myfile:
		myfile.write("{:d}\t{:f}\t{:f}\t{:f}\n".format(x, mag, susc,heatc))
