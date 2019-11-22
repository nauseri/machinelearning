
# practice putting the pieces together and working through a standard machine learning dataset en d -t o -end
# Work through the iris dataset end-to-end

# steps:
# Understanding your data using descriptive statistics and visualization.
# Preprocessing the data to best expose the structure of the problem.
# Spot-checking a number of algorithms using your own test harness.
# Improving results using algorithm parameter tuning.
# Improving results using ensemble methods.
# Finalize the model ready for future use.

# Python version
import sys
print('Python: {}'.format(sys.version))
import scipy
print('scipy: %s' % scipy.__version__)
import numpy as np
print('numpy: %s' % np.__version__)
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
import pandas as pd
print('pandas: %s' % pd.__version__)
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
import sklearn  #scikit-learn
print('sklearn: %s' % sklearn.__version__)
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
import keras
print('keras: %s' % keras.__version__)


from pandas import read_csv
import matplotlib.pyplot as plt
from pandas import set_option
from pandas.plotting import scatter_matrix
from pandas.plotting import hist_frame


# https://archive.ics.uci.edu/ml/datasets/Iris
# Attribute Info: 1. sepal length in cm, 2. sepal width in cm, 3. petal length in cm, 4. petal width in cm,
# 5. class: Iris Setosa, Iris Versicolour, Iris Virginica
filename = 'iris.data.csv'
names = ['sepal len', 'sepal wid', 'petal len', 'petal wid', 'class']
data = read_csv(filename, names=names)
array = data.values
X = array[:, 0:4]
Y = array[:, 4]

# Statistical Summary
print('raw data\n', data.head(4))
np.set_printoptions(precision=4)
print('shape: ', data.shape)
data.info()
class_counts = data.groupby('class').size()
print(class_counts)
# print(data.dtypes)
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print('description\n', description)
correlations = data.corr(method='pearson')
print('correlation\n', correlations)
skew = data.skew()
print('skew\n', skew)   # positive (right) or negative (left) skew

# Scatter Plot Matrix
hist_frame(data, bins=10)
plt.xlabel("nr")
plt.ylabel("frec")
plt.show()
data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()
data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
plt.show()

#  linear and logistic regression can have poor performance if there are
#  highly correlated input variables in your data
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 5, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
scatter_matrix(data)
plt.show()




