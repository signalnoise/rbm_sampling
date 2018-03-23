
path = "T_2.27_epoch_5000_batch_10_hidden_8_ising_8_k_1/rawdata/"

names = ['m','m_error', 'm^2', 'm^2_error']

for i in range(1000, 5010, 10):

	with open(path + "Mag_history." + str(i)) as file:
		li = file.read().splitlines()

	with open("10batches_k1.txt", 'a') as file:
		for item in li:
			file.write(item + "\t")
		file.write("\n")