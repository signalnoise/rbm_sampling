import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec

sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()
sns.set_style('ticks', {"axes.linewidth": ".5", "xtick.minor.size" : ".5", "ytick.minor.size" : ".5","xtick.major.size" : "2", "ytick.major.size" : "2"
	,"xtick.direction" : "in", "ytick.direction" : "in"})
sns.set_context("notebook")


pd.read_csv()