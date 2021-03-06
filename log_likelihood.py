import sys
sys.path.append('../Refactor/')
sys.path.append('../rbm-pytorch-refactor/')
import rbm_pytorch
from torch.autograd import Variable
import torch

def NLL_estimate(rbm, train_loader, n_samples, dtype=torch.FloatTensor):	
	sum_free_energy = 0
	for i, (data, target) in enumerate(train_loader):
		v = Variable(data.view(-1, len(rbm.v_bias.data))).type(dtype)
		sum_free_energy = sum_free_energy + rbm.free_energy(v, size_average=False).sum().data[0]
	logZ, highZ, lowZ = rbm.annealed_importance_sampling()
	avg_f_e = sum_free_energy/n_samples
	nll = logZ + avg_f_e
	upper_bound = (highZ) + avg_f_e
	lower_bound = (lowZ) + avg_f_e
	return (nll, upper_bound, lower_bound)

def LL_estimate(rbm, train_loader, n_samples):
	sum_free_energy = 0
	for i, (data, target) in enumerate(train_loader):
		v = Variable(data.view(-1, self.args.visible_n))
		sum_free_energy = sum_free_energy + rbm.free_energy(v, size_average=False).sum().data[0]
	logZ, highZ, lowZ = rbm.annealed_importance_sampling()
	avg_f_e = sum_free_energy/n_samples
	ll = -logZ - avg_f_e
	upper_bound = -(highZ) - avg_f_e
	lower_bound = -(lowZ) - avg_f_e
	return (ll, upper_bound, lower_bound)