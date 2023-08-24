#!/usr/bin/python3

"""
Assignment: CS5014-P1
Author: 220025456
machine learning logistic regression model that can predict bean type.

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
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score, PrecisionRecallDisplay, precision_recall_curve
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

print("1. Loading data...")

# read data from csv file
try:
    data = pd.read_csv("drybeans.csv")
except FileNotFoundError:
    print("File not found. Please check the path to the dataset file.")
    sys.exit(1)

print("2. Performing Data cleaning...")
print("2a. dropping missing values...")
# drop rows having missing values, if any
data.dropna(axis=0, how='any', inplace=True)

print("2b. dropping duplicates...")
# 68 duplicate row identified. drop duplicate rows
data.drop_duplicates(keep='first', inplace=True)

print("3. Assigning numerical values to class labels...")
# assign numerical values to class labels.
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

# identify features and label cols
X, y = data.iloc[:, :-1], data.iloc[:, -1]

print("4. Splitting data into training and test sets...")
# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("5. Normalizing training set...")
print("6. Normalizing test set...")
# Normalize input data
scaler = preprocessing.StandardScaler()
X_train[:] = scaler.fit_transform(X_train)
X_test[:] = scaler.transform(X_test)

global model  # global model variable


def modelling(features_train, label_train, penalty='none', class_weight=None):
    """
    :param features_train: training features
    :param label_train: training labels
    :param penalty: penalty type
    :param class_weight: class weight
    :return: None
    description: train a logistic regression model
    """
    global model
    print("Training and Evaluating model with penalty = {} and class_weight = {}...".format(penalty, class_weight))
    model = LogisticRegression(penalty=penalty, class_weight=class_weight, max_iter=100).fit(features_train, label_train)


def evaluate_model(features_test, label_test):
    """
    :param features_test: test features
    :param label_test: output labels
    :return: prints model evaluation metrics
    """
    y_predicted = model.predict(features_test)
    print('\tpredict() = {}\n'.format(y_predicted))
    print('\tdecision_function(). Showing first 5 rows')
    [print(line) for line in model.decision_function(features_test)[:5]]
    print()
    print('\tpredict_proba(). Showing first 5 rows')
    [print(line) for line in model.predict_proba(features_test)[:5]]
    print()
    print('\tConfusion Matrix')
    print(confusion_matrix(label_test, y_predicted))
    print()
    print('\tAccuracy score = {}'.format(accuracy_score(label_test, y_predicted)))
    print('\tBalanced accuracy score = {}'.format(balanced_accuracy_score(label_test, y_predicted)))
    print('\tPrecision score (micro) = {}'.format(precision_score(label_test, y_predicted, average='micro')))
    print('\tRecall score (micro) = {}'.format(recall_score(label_test, y_predicted, average='micro')))
    print('\tPrecision score (macro) = {}'.format(precision_score(label_test, y_predicted, average='macro')))
    print('\tRecall score (macro) = {}'.format(recall_score(label_test, y_predicted, average='macro')))
    print('\tF1 score (micro) = {}'.format(f1_score(label_test, y_predicted, average='micro')))
    print('\tF1 score (macro) = {}'.format(f1_score(label_test, y_predicted, average='macro')))
    print()
    print('-' * 100)
    print()


# implement 2nd degree polynomial features
def polynomial_features(penalty='none', class_weight=None):
    """
    :param penalty: none or l2
    :param class_weight: None or balanced
    :return: none
    description: train and evaluate model for 2nd degree polynomial features.
    """
    print("Implementing 2nd degree polynomial features...")
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.fit_transform(X_test)
    print('2nd degree polynomial dimension = {}'.format(X_poly_train.shape[1]))
    modelling(X_poly_train, y_train, penalty, class_weight)
    evaluate_model(X_poly_test, y_test)


# PR curve
def precision_recall_plot(features_test, label_test):
    """
    :param features_test: X_test_best input
    :param label_test: y_test labels
    :return: displays precision recall plot
    """
    _, ax = plt.subplots(figsize=(7, 8))
    colors = ["navy", "turquoise", "darkorange", "cornflowerblue", "black", "red", "green"]
    for i in range(7):
        precision, recall, _ = precision_recall_curve(label_test, model.decision_function(features_test)[:, i],
                                                      pos_label=i)
        display = PrecisionRecallDisplay(
            recall=recall,
            precision=precision
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=colors[i])

    ax.set_title("L2-Balanced Precision-Recall curve")
    plt.legend(loc="best").set_draggable(True)
    plt.show()


# correlated features
def correlated_features(train, test):
    """
    :param train: training features
    :param test: test features
    :param threshold: threshold for correlation
    :return: highly correlated features dropped from train and test.
    description: check for highly correlated features and drop them.
    """
    drop_cols = ['Perimeter', 'EquivDiameter', 'ConvexArea']
    print("7. drop highly correlated features {}".format(drop_cols))
    print('-' * 100, end='\n\n')
    return train.drop(drop_cols, axis=1), test.drop(drop_cols, axis=1)


parm = {'Unbalanced': ['none', None], 'Balanced': ['none', 'balanced'],
        'L2-Balanced': ['l2', 'balanced'], 'Poly_Unbalanced': ['none', None], 'Poly_Balanced': ['l2', None]}

X_train_best, X_test_best = correlated_features(X_train, X_test)

for key, value in parm.items():
    if 'Poly' in key:
        polynomial_features(value[0], value[1])
    else:
        modelling(X_train_best, y_train, value[0], value[1])
        evaluate_model(X_test_best, y_test)
        if key == 'L2-Balanced':
            precision_recall_plot(X_test_best, y_test)
