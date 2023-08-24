"""
Assignment: CS5011-A4
Author: 220025456
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import modeller
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
    

def train(model, criterion, train_loader, validation_loader, optimizer, model_type, epochs=5 ,plot=False):
    LOSS=[]
    
    for epoch in range(epochs):
        train_loss=0
        val_loss  =0
        
        for x,y in train_loader:
            model.train()
            #make a prediction 
            yhat=model(x)
            #calculate the loss
            loss=criterion(yhat,y)

            if model_type == 'classification_nn_cost':
                regret = torch.min(yhat, dim=1).values - torch.min(y, dim=1).values # regret = min(yhat) - min(y)
                loss = torch.mean(torch.mean(loss, 1) * torch.abs(regret)) # loss = mean(mean(loss) of instance * regret)

            train_loss+=loss.item()
            #clear gradient 
            optimizer.zero_grad()
            #Backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            #the step function on an Optimizer makes an update to its parameters
            optimizer.step()
            
        #store loss for each epoch
        LOSS.append(avg_loss := train_loss/len(train_loader))

        correct = 0
        for xVal, yVal in validation_loader:
            model.eval()
            yhat_val = model(xVal)
            vloss = criterion(yhat_val, yVal)

            if model_type == 'classification_nn_cost':
                regret = torch.min(yhat_val, dim=1).values - torch.min(yVal, dim=1).values
                vloss = torch.mean(torch.mean(vloss, 1) * torch.abs(regret))

            val_loss += vloss.item()
            if model_type == 'classification_nn':
                _, yidx = torch.max(yhat_val.data, 1)
                correct += (yidx == yVal).sum().item()
        
        if epoch % 100 == 0:
            print(f'Training avg loss for epoch {epoch}: {avg_loss:0.2e}')
            print(f'Validation avg loss for epoch {epoch}: {val_loss/len(validation_loader):0.2e}')
            if model_type == 'classification_nn':
                print(f'Validation accuracy for epoch {epoch}: {correct/len(validation_loader.dataset):0.2f}')
    
    if plot:
        plt.figure()
        plt.plot(LOSS)
        plt.xlabel('epoch')
        plt.ylabel('LOSS')
        plt.show()
    return LOSS


def train_val_split(x, y, k, classification=False):
    x_train = None
    for i in range(5):
        if i==k:
            x_val = x[i]
            y_val = y[i]
        elif x_train is None:
            x_train = x[i]
            y_train = y[i]
        else:
            x_train = np.vstack((x_train, x[i]))
            if classification:
                y_train = np.hstack((y_train, y[i]))
            else:
                y_train = np.vstack((y_train, y[i]))

    if classification:
        tset = modeller.classification_Data(x_train, y_train)
        vset = modeller.classification_Data(x_val, y_val)
    else:
        tset = modeller.Data(x_train, y_train)
        vset = modeller.Data(x_val, y_val)        
    
    return tset, vset


def main():
    parser = argparse.ArgumentParser(description="Train an AS model and save it to file")
    parser.add_argument("--model-type", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    parser.add_argument("--save", type=str, required=True, help="Save the trained model (and any related info) to a .pt file")
    
    args = parser.parse_args()

    print(f"\nTraining a {args.model_type} model on {args.data}, and save it to {args.save}")
    
    # YOUR CODE HERE
    # Load data
    x = np.loadtxt(args.data + "instance-features.txt")
    y = np.loadtxt(args.data + "performance-data.txt")

    # check for missing values
    print(f"Missing values in training set {np.isnan(x).sum()}, {np.isnan(y).sum()}")

    # check for duplicate rows
    _, dup = np.unique(x, return_counts=True, axis=0)
    print("Any duplicate samples", len(np.argwhere(dup > 1)))

    stats = []
    config = {'layer': 3, 'hidden': 100, 'epochs': 1000, 'batch_size': 50, 'lr': 0.001, 'loss': 0.0}
    classification = False

    # feature selection based on gini importance
    print(f"Performing feature selection...")
    best_features = modeller.feature_selection(x, y)
    best_features = np.flatnonzero(best_features) #115 features
    x1 = x[:, best_features]
    torch.save(best_features, args.save[ : args.save.rindex('/')+1] + 'best_features.pt')

    input  = x1.shape[1]
    output = y.shape[1]
    hidden = config['hidden']

    if args.model_type == 'regresion_nn':
        model = modeller.regression_nn(input, hidden, output)
        criterion = nn.MSELoss()
    elif args.model_type == 'classification_nn':
        classification = True
        y = np.argmin(y, axis=1)    # convert to class labels
        model = modeller.classification_nn(input, hidden, output)
        criterion = nn.CrossEntropyLoss()
    elif args.model_type == 'classification_nn_cost':
        model = modeller.regression_nn(input, hidden, output)
        criterion = nn.MSELoss(reduction='none')
    elif args.model_type == 'random_forest':
        forest = modeller.random_forest(x1, y)
        print(f"Saving the trained model to {args.save}")
        torch.save(forest, args.save)
        return   

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.0001)

    # split data into 5 folds
    X = np.array_split(x1, 5)
    Y = np.array_split(y, 5)

    # for each fold, train, validate and print stats
    for k in range(5):
        print(f"\n{'-'*50}")
        print(f"Training on fold {k}")
        train_set, val_set = train_val_split(X, Y, k, classification)
    
        train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=config['batch_size'], shuffle=True)
    
        loss=train(model, criterion, train_loader, val_loader, optimizer, args.model_type, epochs=config['epochs'], plot=True)
        config['loss'] = (format(min(loss),'.2e'), np.argmin(loss))
        stats.append(config)
        print(*stats, sep='\n')
        stats.clear()

    print(f"\nTraining finished")
    print(f"Saving the trained model to {args.save}")
    torch.save(model.state_dict(), args.save)


if __name__ == "__main__":
    main()
