import sys
try:
	sys.path.append('../rbm-pytorch-refactor/')
except IOError as e:
	print('Oops')
import rbm_pytorch
import ising_methods_new
import argparse
import json
import torch

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

rbm = rbm_pytorch.RBM(n_vis=parameters['ising']['size']**2, n_hid=parameters['n_hid'])

rbm.load_state_dict(torch.load(parameters['saved_state']))

states = ising_methods_new.sample_from_rbm(rbm, parameters)

print(ising_methods_new.ising_observables(states, 8, 2.27))