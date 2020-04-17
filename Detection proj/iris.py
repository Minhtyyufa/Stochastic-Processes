import scipy.io as sio
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# load the data from matlab mat file
iris = sio.loadmat("Iris.mat")
y = iris['labels']
X = iris['features']

def L(pdf, j, y):
    return pdf[j].pdf(y)/pdf[1].pdf(y)


# Defines the inequalities that classify which group the observation is in, then returns that group. 
# 3-ary MPE Classifier with MAP Rule
def classify(pdf, P1, P2, P3, y):
    L2 = L(pdf, 2, y)
    L3 = L(pdf, 3, y)
    
    # Equations 8.96-8.98
    comp12 = P2*L2 > P1
    comp23 = -P2*L2 > -P3*L3
    comp31 = 0 > -P1 + P3*L3
    
    if not comp31 and comp23:
        return 3
    if comp31 and not comp12:
        return 1
    if comp12 and not comp23:
        return 2
    
# split into test and train 50 to 50
skf = StratifiedKFold(n_splits=2, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # find the prior probability
    P1 = np.count_nonzero(y_train == 1)/len(y_train)
    P2 = np.count_nonzero(y_train == 2)/len(y_train)
    P3 = np.count_nonzero(y_train == 3)/len(y_train)
    
    data = {}
    pdf = {}
    
    labels = np.unique(y_train)
    
    for label in labels:
        data[label] = []
    
    for i in range(len(X_train)):
        data[y_train[i][0]].append(X_train[i])
    
    # find the conditional pdf for likelihood
    for label in labels:
        data[label] = np.array(data[label])
        size = len(data[label][0])
        label_pdf = multivariate_normal(mean=np.mean(data[label], axis=0), cov=np.cov(np.transpose(data[label])))
        pdf[label] = label_pdf
    
    # use m-ary classifier with MAP rule
    y_hat = [[classify(pdf, P1, P2, P3, X_test[i])] for i in range(len(X_test))]
    print((len(y_hat)-np.sum(y_hat == y_test))/len(y_hat))
    print(confusion_matrix(y_hat, y_test))
    
    # plot the confusion matrix
    labels = range(1,4)
    sns.heatmap(confusion_matrix(y_hat,y_test), cmap="YlGnBu", annot=True, fmt="d", yticklabels=labels, xticklabels=labels)
    plt.title('Confusion Matrix for Iris Classifier')
    plt.show()
    

    
    