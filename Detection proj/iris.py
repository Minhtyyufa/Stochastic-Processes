import scipy.io as sio
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import StratifiedKFold

iris = sio.loadmat("Iris.mat")
y = iris['labels']
X = iris['features']

skf = StratifiedKFold(n_splits=2, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    data = {}
    pdf = {}
    for label in np.unique(y_train):
        data[label] = []
    
    for i in range(len(X_train)):
        data[y_train[i][0]].append(X_train[i])
    
    for label in np.unique(y_train):
        data[label] = np.array(data[label])
        size = len(data[label][0])
        label_pdf = multivariate_normal(mean=np.mean(data[label], axis=0), cov=np.cov(np.transpose(data[label])))
        pdf[label] = label_pdf
    
    