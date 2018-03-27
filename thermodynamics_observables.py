import sys
sys.path.append('../rbm-pytorch-refactor/')
import rbm_pytorch
import rbm_interface
import ising_methods_new
import argparse
from torch.utils.data import DataLoader
import json
import torch

def sample_and_save(temperature, saved_state, parameters, filename, dtype):
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
	
	mag, susc, energy, heatc = ising_methods_new.ising_observables(states, L, temperature)

	with open(filename, "a") as file:
		file.write("{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\n".format(temperature, 
					mag[0], mag[1], susc[0], susc[1], energy[0], energy[1], heatc[0], heatc[1]))

parse = argparse.ArgumentParser(description='Process some integers.')
parse.add_argument('--json', dest='input_json', default='params.json', help='JSON file describing the sample parameters',
					type=str)
parse.add_argument('--input_path', dest='input_path', help='Path to trained rbms')
parse.add_argument('--output_path', dest='output_path', help='Path to output data')
parse.add_argument('--training_data', dest='training_data', help='Path to training data')
parse.add_argument('--cuda', dest='cuda', type=bool)

args = parse.parse_args()

# Enable cuda
if args.cuda:
	dtype = torch.cuda.FloatTensor
else:
	dtype = torch.FloatTensor

# Load json params
try:
	parameters = json.load(open(args.input_json))
except IOError as e:
	print("I/O error({0}) (json): {1}".format(e.errno, e.strerror))
	sys.exit(1)
except:
	print("Unexpected error:", sys.exc_info()[0])
	raise

# Loop over epochs and append data to files
for i in range(1, 31, 1):

	input_dir = args.input_path + "/8-" + str(i) + "/" 

	image_dir = input_dir + "/images/"

	training_data = "/exports/csce/eddie/ph/groups/rbm_ml/owen-data/training_data/state" + str(i) + ".txt"

	with open(training_data) as file:
		temperature = float(file.readline())

	data, validation, comparison = rbm_interface.ising_loader(training_data, size=64).get_datasets()
	train_loader = DataLoader(data, shuffle=True, batch_size=100000, drop_last=True)
	for i, (data, target) in enumerate(train_loader):
		mag, susc, energy, heatc = ising_methods_new.ising_observables(data, parameters['ising']['size'], temperature)

	with open(args.input_path + "training_data_observables.txt", "a") as file:
		file.write("{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\n".format(temperature, 
					mag[0], mag[1], susc[0], susc[1], energy[0], energy[1], heatc[0], heatc[1]))

	saved_state = input_dir + "/trained_rbm.pytorch.3000"
	filename = args.input_path + "temp_graph_machines.txt"
	sample_and_save(temperature, saved_state, parameters, filename, dtype)
