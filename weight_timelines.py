import sys
sys.path.append('../rbm-pytorch-refactor/')
sys.path.append('../Refactor')
import rbm_pytorch
import rbm_interface
import ising_methods_new
import argparse
from torch.utils.data import DataLoader
import json
import torch
from log_likelihood import NLL_estimate
from ising_methods_new import imgshow
from torchvision.utils import make_grid

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
parse.add_argument('--n_hid', dest='n_hid', type=int, help='Path to output data')
parse.add_argument('--training_data', dest='training_data', help='Path to training data')
parse.add_argument('--state_number', dest='state_number', help='Which state are we using')
parse.add_argument('-L', dest='lattice_size', type=int, help='Ising lattice size')
parse.add_argument('--cuda', dest='cuda', type=bool)

args = parse.parse_args()

state_number = args.state_number
L = args.lattice_size

# Enable cuda
if args.cuda:
	dtype = torch.cuda.FloatTensor
else:
	dtype = torch.FloatTensor


input_dir = args.input_path + "/8-" + str(state_number) + "/" 

#training_data = args.training_data + "/state" + str(state_number) + ".txt"

#data, validation, comparison = rbm_interface.ising_loader(training_data, size=L**2).get_datasets()
#train_loader = DataLoader(data, shuffle=True, batch_size=1000, drop_last=True)
#file = open(input_dir + "weight_timeline.data", 'w')
#file.write("# Sample temperature " + str(temperature))
# Loop over epochs and append data to files
for epoch in range(0, 3001, 10):
	
	saved_state = input_dir + "/trained_rbm.pytorch." + str(epoch).zfill(4)

	rbm = rbm_pytorch.RBM(n_vis=L**2, n_hid=args.n_hid, enable_cuda=args.cuda)
	rbm.load_state_dict(torch.load(saved_state)) #map_location=lambda storage, loc: storage))

	imgshow(input_dir + "/weights/" + str(epoch).zfill(5), make_grid(rbm.W.view(64, 1, L, L).data))
	imgshow(input_dir + "/biases/" + str(epoch).zfill(5), make_grid(rbm.v_bias.view(-1, 1, L, L).data))
	imgshow(input_dir + "/ciases/" + str(epoch).zfill(5), make_grid(rbm.h_bias.view(-1, 1, L, L).data))
	#file.write("{:d}\t{:f}\t{:f}\t{:f}\n".format(epoch, avg_weight, avg_bias, avg_cias))
	print(str(epoch))
file.close()
