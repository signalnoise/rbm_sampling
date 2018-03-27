import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import seaborn as sns
import numpy as np

from scipy.stats import norm

def bootstrap_err(nd_array, N_bootstrap):
	indices = np.random.randint(len(nd_array), size = (N_bootstrap, len(nd_array)))
	bootstrap = np.take(nd_array,indices)
	err = np.sqrt(np.var(np.var(bootstrap, axis=1)))
	return err

font = {'weight' : 'light',
        'size'   : 10}

matplotlib.rc('font', **font)
#matplotlib.rc('text', usetex=True)

sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()
sns.set_style('white')
sns.set_style('ticks',{"axes.linewidth": ".5", "xtick.minor.size" : ".5", "ytick.minor.size" : ".5","xtick.major.size" : "3", "ytick.major.size" : "3"})
sns.set_context('paper')

path = "batchsizes/"

names = ['m','m_error', 'm^2', 'm^2_error']
labels = ['N$_{batch}$=100', 'N$_{batch}$=50', 'N$_{batch}$=10']
N_bootstrap = 100000

df = pd.read_csv(path + "100batches_k1.txt", delim_whitespace=True, names = names)
df2 = pd.read_csv(path + "50batches_k1.txt", delim_whitespace=True, names = names)
df3 = pd.read_csv(path + "10batches_k1.txt", delim_whitespace=True, names = names)

var = df.as_matrix(columns=['m^2'])/(2.27)
var2 = df2.as_matrix(columns=['m^2'])/(2.27)
var3 = df3.as_matrix(columns=['m^2'])/(2.27)

print(var.std())
print(var2.std())
print(var3.std())


print(bootstrap_err(var, N_bootstrap))
print(bootstrap_err(var2, N_bootstrap))
print(bootstrap_err(var3, N_bootstrap))


'''
sns.distplot(var, fit=norm, kde=False, fit_kws={"color": palette[0]})
sns.distplot(var2, fit=norm, kde=False, fit_kws={"color": palette[1]})
sns.distplot(var3, fit=norm, kde=False, fit_kws={"color": palette[2]})
plt.axvline(1.21, color='k', ls='--', linewidth=0.5)
plt.xlabel('$\chi$')
plt.ylabel('Frequency Density')
plt.legend(labels)
#print(df)
plt.show()
#plt.savefig('test.png')
'''