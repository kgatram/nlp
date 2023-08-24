"""
Assignment: CS5014-P2
Author: 220025456
machine learning image classification of seals.

@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
"""

import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings('ignore')

global hog_pca
global nd_pca
global rgb_pca
jobs = -1

try:
    inmode = sys.argv[1]                # 'binary' or 'multi'
    feature_selection = sys.argv[2]     # True or False
except IndexError:
    print('Usage: python seals.py <binary|multi> <True|False>')
    sys.exit(1)

path = "/data/cs5014/P2/" + inmode + "/"

# Load binary data
X_train = pd.read_csv(path + "X_train.csv", header=None)
Y_train = pd.read_csv(path + "Y_train.csv", header=None)
X_test = pd.read_csv(path + "X_test.csv", header=None)

x1_train, x1_test, y1_train, y1_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42,
                                                        stratify=Y_train)


def perform_PCA(X, subset):
    # Perform dimensionality reduction using PCA
    threshold = 0.95
    pca = PCA()
    pca.fit(X)
    var = pca.explained_variance_ratio_
    n_component = np.argmax(np.cumsum(var) >= threshold) + 1  # first n components having 95% variance in given data.
    print(f'No. of {subset} features giving %.0d%% of variance = {n_component}' % (threshold * 100))

    plt.plot(np.cumsum(var))
    plt.xlabel('No of ' + subset + ' features')
    plt.ylabel('Cumulative variance of ' + subset)
    plt.title('PCA Analysis')
    plt.show()

    return pca


def reduce_dim(X, fit=False):
    global hog_pca
    global nd_pca
    global rgb_pca

    hog = X.iloc[:, :900]
    nd = X.iloc[:, 900:916]
    rgb = X.iloc[:, 916:]

    if fit:
        hog_pca = perform_PCA(hog, 'HOG')
        nd_pca = perform_PCA(nd, 'ND')
        rgb_pca = perform_PCA(rgb, 'RGB')

    hog = hog_pca.transform(hog)
    nd = nd_pca.transform(nd)
    rgb = rgb_pca.transform(rgb)

    return np.concatenate((hog, nd, rgb), axis=1)


def modelling(model, xtrain, ytrain, xtest, ytest=None, mode='predict'):
    if model in 'rfc':
        print(f'{mode.capitalize()}ing Random Forest')
        pipe = make_pipeline(RandomForestClassifier(n_jobs=jobs))
    elif model in 'svm':
        print(f'{mode.capitalize()}ing SVM')
        pipe = make_pipeline(StandardScaler(), SVC())
    elif model in 'knn':
        print(f'{mode.capitalize()}ing KNN')
        pipe = make_pipeline(KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=jobs))
    elif model in 'nb':
        print(f'{mode.capitalize()}ing Naive Bayes')
        pipe = make_pipeline(StandardScaler(), GaussianNB())

    pipe.fit(xtrain, np.ravel(ytrain))
    predicted = pipe.predict(xtest)
    for label in np.unique(ytrain):
        count = np.where(predicted == label, 1, 0).sum()
        print(f'No. of {label} predicted = {count}')

    if mode in 'evaluat':
        print(classification_report(ytest, predicted, zero_division=0))
    else:
        fname = inmode + '/Y_test.csv'
        print(f'Saving file as {fname}')
        np.savetxt(fname, predicted, fmt='%s', delimiter=',')


# def tunning():
#     param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf']}
#     grid = GridSearchCV(SVC(),param_grid,refit=False,verbose=2,n_jobs=jobs)
#     grid.fit(X_train,y_train)
#     print(grid.best_estimator_)


classifiers = ['rfc', 'svm', 'knn', 'nb']

if feature_selection:
    X_train = reduce_dim(X_train, True)

    x1_train = reduce_dim(x1_train, False)
    x1_test = reduce_dim(x1_test, False)

for model in classifiers:
    modelling(model, x1_train, y1_train, x1_test, y1_test, 'evaluat')

# predict with best model
modelling('svm', X_train, Y_train, X_test, None, 'predict')
exit(0)
