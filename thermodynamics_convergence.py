import sys
sys.path.append('../rbm-pytorch-refactor/')
import rbm_pytorch
import ising_methods_new
import argparse

import json
import torch

def sample_and_save(temperature, saved_state, parameters, filename, epoch, dtype):
	""" Appends the four observables for a given trained rbm to a file.
	Args:
		n_vis: number of visible nodes
		n_hid: number of hidden nodes
		temperature: temperature of the training data.
		saved_state: torch save file for the rbm.
		parameters: json parameters for sampling.
		filename: path to where data should be saved.
	"""
	L = parameters['ising']['size']
	rbm = rbm_pytorch.RBM(n_vis=L**2, n_hid=parameters['n_hid'])
	rbm.load_state_dict(torch.load(saved_state))

	states = ising_methods_new.sample_from_rbm(rbm, parameters, dtype)
	
	with open(filename, "a") as file:
		myfile.write("{:d}\t{:f}\t{:f}\t{:f}\n".format(epoch,ising_methods_new.ising_observables(states, L, temperature)))

parse = argparse.ArgumentParser(description='Process some integers.')
parse.add_argument('--json', dest='input_json', default='params.json', help='JSON file describing the sample parameters',
					type=str)
parse.add_argument('--input_path', dest='input_path', help='Path to trained rbms')
parse.add_argument('--output_path', dest='output_path', help='Path to output data')
parse.add_argument('--training_data', dest='training_data', help='Path to training data')
parse.add_argument('--cuda', dest='cuda', type=bool)

args = parse.parse_args()

if args.cuda:
	dtype = torch.cuda.FloatTensor
else:
	dtype = torch.FloatTensor

with open(args.training_data) as file:
	temperature = float(file.readline())

try:
	parameters = json.load(open(args.input_json))
except IOError as e:
	print("I/O error({0}) (json): {1}".format(e.errno, e.strerror))
	sys.exit(1)
except:
	print("Unexpected error:", sys.exc_info()[0])
	raise

for e in range(0, 10, parameters['epochs'] + 10):

	saved_state = args.input_path + "/trained_rbm.pytorch." + str(e)
	filename = str(temperature) + "_thermo_convergence.data"
	sample_and_save(temperature, saved_state, parameters, filename, e)
