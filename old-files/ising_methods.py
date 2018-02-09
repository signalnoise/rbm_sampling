import numpy as np
import torch



# Class allowing for a little easier state by state calculation of thermodynamic properties
class IsingState:

	def __init__(self, state, ising_size_L):

		self.shape = state.shape
		state = state.numpy()
		self.mag = np.mean(np.add(np.multiply(state,2),-1))*ising_size_L**2
		self.state = state.reshape((ising_size_L, ising_size_L))

	def energy(self):

		# Takes the product of every spin with its bottom and right neighbours
		e = 0
		for i in range(self.state.shape[0]):
			for j in range(self.state.shape[1]):
				e = e - self.spin(i, j)*(self.right(i, j) + self.below(i, j))
		return e

	def spin(self, i, j):

		# Convert spins from 0/1 to -1/1
		return 2*self.state[i, j] - 1

	# These methods implement periodic boundary conditions
	def right(self, i, j):

		if i == self.state.shape[0] - 1:
			index = 0
		else:
			index = i + 1
		return self.spin(index, j)

	def below(self, i, j):

		if j == self.state.shape[1] - 1:
			jndex = 0
		else:
			jndex = j + 1
		return self.spin(i, jndex)


def magnetisation(state, size):
	return abs(np.mean(np.add(np.multiply(state, 2), -1)) * size)

def heat_capacity(energies, temperature, size):

	return np.var(energies, dtype="Float64")/(size ** 2 * temperature ** 2)


def avg_magnetisation(states,model_size):

	magnetisations = np.zeros(len(states))
	for i in range(len(states)):
		magnetisations[i] = np.absolute(magnetisation(states[i,:],model_size))
	return np.mean(magnetisations, dtype="Float64")/model_size

