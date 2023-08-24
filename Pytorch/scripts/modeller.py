"""
Assignment: CS5011-A4
Author: 220025456
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


 # Create Data Class

class Data(Dataset):
    
    # Constructor
    def __init__(self, feature, label):
        self.x = torch.tensor(feature, dtype=torch.float32)
        self.y = torch.tensor(label, dtype=torch.float32)
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self,index):    
        return self.x[index,:],self.y[index,:]
    
    # Get Length
    def __len__(self):
        return self.len
    

class classification_Data(Dataset):
    
    # Constructor
    def __init__(self, feature, label):
        self.x = torch.tensor(feature, dtype=torch.float32)
        self.y = torch.tensor(label, dtype=torch.int64)
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self,index):    
        return self.x[index,:],self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len


def regression_nn(input, hidden, output):
    model= torch.nn.Sequential(
        torch.nn.BatchNorm1d(input),
        torch.nn.Linear(input, hidden), #1
        torch.nn.BatchNorm1d(hidden),
        torch.nn.ReLU(), 
        torch.nn.Linear(hidden, hidden), #2
        torch.nn.BatchNorm1d(hidden),
        torch.nn.ReLU(), 
        torch.nn.Linear(hidden, output) #3
    )
    return model


def classification_nn(input, hidden, output):
    model= torch.nn.Sequential(
        torch.nn.BatchNorm1d(input),
        torch.nn.Linear(input, hidden), #1
        torch.nn.BatchNorm1d(hidden),
        torch.nn.ReLU(), 
        torch.nn.Linear(hidden, hidden), #2
        torch.nn.BatchNorm1d(hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, output) #3
    )
    return model


def feature_selection(x, y):
    regr_rf = RandomForestRegressor(n_estimators=100, random_state=2)
    regr_rf.fit(x, y)
    best_features = regr_rf.feature_importances_
    return best_features


def random_forest(x, y):
    forest = []
    y1 = np.argmin(y, axis=1) # convert to class labels
    z = np.column_stack((x, y1))
    print('Generating random forest...')

# create a random forest for each pair of algorithm 
    for i in range(y.shape[1]):
        for j in range(y.shape[1]):
            if i < j:
                clf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=2)
                input = z[(z[:,-1] == i) | (z[:,-1] == j)][:, :-1] # features
                label = z[(z[:,-1] == i) | (z[:,-1] == j)][:, -1] # class labels
                clf.fit(input, label)
                forest.append(clf)
    
    return forest