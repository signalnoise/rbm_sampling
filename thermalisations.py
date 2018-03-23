import sys
sys.path.append('../Refactor/')
import rbm_pytorch
import ising_methods_new
import argparse
import json
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

font = {'family' : 'normal',
        'weight' : 'light',
        'size'   : 10}

matplotlib.rc('font', **font)

sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()

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

rbm.load_state_dict(torch.load(parameters['saved_state'], map_location=lambda storage, loc: storage))
x = np.array([[0,1,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,0,0,1,1]])
vin = torch.from_numpy(x)

for i in range(6,7,1):
	parameters['thermalisation'] = 10**i
	states = ising_methods_new.sample_from_rbm(rbm, parameters, v_in=vin, save_images=True)


