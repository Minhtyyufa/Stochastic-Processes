import scipy.io as sio
iris = sio.loadmat("Iris.mat")
iris_labels = iris['labels']
iris_features = iris['features']
