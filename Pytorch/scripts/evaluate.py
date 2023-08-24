"""
Assignment: CS5011-A4
Author: 220025456
"""

import argparse
import numpy as np
import modeller
import torch
from torch import nn


# Part 1 evaluation
def regression(data_path, model_path):
    config = {'h-layer': 3, 'hidden': 100}

    # Load data
    x = np.loadtxt(data_path + "instance-features.txt")
    y = np.loadtxt(data_path + "performance-data.txt")
    n_samples = x.shape[0]

    best_features = torch.load(model_path[ : model_path.rindex('/')+1] + "best_features.pt")
    x1 = x[:, best_features]

    test_set = modeller.Data(x1, y)

    # Load model
    model = modeller.regression_nn(x1.shape[1], config['hidden'], y.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    yhat = model(test_set.x)
    avg_loss = torch.mean((yhat - test_set.y)**2).item()

    # Given an instance, choose the algorithm with the lowest predicted cost.
    correct = 0
    for i in range(y.shape[0]):
        correct += (np.argmin(yhat[i].detach().numpy()) == np.argmin(test_set.y[i].detach().numpy())).sum()
    accuracy = correct / n_samples

    # Calculate the average cost of the predicted algorithms on the given dataset
    # avg_cost = torch.min(torch.mean(yhat, 0)).item() 
    avg_cost = torch.mean(torch.min(yhat, dim=1).values).item()

    # Calculate the average cost of the SBS and the VBS
    sbs_avg_cost = torch.min(torch.mean(test_set.y, 0)).item() # the algorithm with the best average performance on known instances
    vbs_avg_cost = torch.mean(torch.min(test_set.y, dim=1).values).item() # the algorithm with the best performance on each instance

    sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)

    print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")


# Part 2 - basic evaluation
def classification(data_path, model_path):
    config = {'h-layer': 3, 'hidden': 100}

    # Load data
    x = np.loadtxt(data_path + "instance-features.txt")
    y = np.loadtxt(data_path + "performance-data.txt")
    n_samples = x.shape[0]
    output = y.shape[1]

    best_features = torch.load(model_path[ : model_path.rindex('/')+1] + "best_features.pt")
    x1 = x[:, best_features]

    y1 = np.argmin(y, axis=1)    # convert to class labels

    test_set = modeller.classification_Data(x1, y1)

    # Load model
    model = modeller.classification_nn(x1.shape[1], config["hidden"], output)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    yhat = model(test_set.x)
    criterion = nn.CrossEntropyLoss()

    avg_loss = torch.mean(criterion(yhat, test_set.y)).item()

    correct = 0
    _, yidx = torch.max(yhat.data, 1)
    correct += (yidx == test_set.y).sum().item()
    accuracy = correct / n_samples

    # Calculate the average cost of the predicted algorithms on the given dataset
    avg_cost=0
    for i in range(len(yidx)):
        avg_cost += y[i, yidx[i]]
    avg_cost = avg_cost/len(yidx)

    # Calculate the average cost of the SBS and the VBS
    sbs_avg_cost = np.min(np.mean(y, axis=0))  # the algorithm with the best average performance on known instances
    vbs_avg_cost = np.mean(np.min(y, axis=1))  # the algorithm with the best performance on each instance

    sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)

    print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")


# Part 3 - advance evaluation
def classification2(data_path, model_path):
    config = {'h-layer': 3, 'hidden': 100}

    # Load data
    x = np.loadtxt(data_path + "instance-features.txt")
    y = np.loadtxt(data_path + "performance-data.txt")
    best_features = torch.load(model_path[ : model_path.rindex('/')+1] + "best_features.pt")
    n_samples = x.shape[0]

    x1 = x[:, best_features]

    test_set = modeller.Data(x1, y)

    # Load model
    model = modeller.regression_nn(x1.shape[1], config['hidden'], y.shape[1])
    model.load_state_dict(torch.load(model_path))
    criterion = nn.MSELoss(reduction='none')
    model.eval()
    yhat = model(test_set.x)

    avg_loss=criterion(yhat,test_set.y)
    regret = torch.min(yhat, dim=1).values - torch.min(test_set.y, dim=1).values # regret = min(yhat) - min(y)
    avg_loss = torch.mean(torch.mean(avg_loss, 1) * torch.abs(regret)) # loss = mean(mean(loss) of instance * regret)

    # Given an instance, choose the algorithm with the lowest predicted cost.
    correct = 0
    for i in range(y.shape[0]):
        correct += (np.argmin(yhat[i].detach().numpy()) == np.argmin(test_set.y[i].detach().numpy())).sum()
    accuracy = correct / n_samples

    # Calculate the average cost of the predicted algorithms on the given dataset
    # avg_cost = torch.min(torch.mean(yhat, 0)).item() 
    avg_cost = torch.mean(torch.min(yhat, dim=1).values).item()

    # Calculate the average cost of the SBS and the VBS
    sbs_avg_cost = torch.min(torch.mean(test_set.y, 0)).item() # the algorithm with the best average performance on known instances
    vbs_avg_cost = torch.mean(torch.min(test_set.y, dim=1).values).item() # the algorithm with the best performance on each instance

    sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)

    print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")


# Part 4 - random forest pair-wise classifier 
def random_forest(data_path, model_path):
    
    # Load data
    x = np.loadtxt(data_path + "instance-features.txt")
    y = np.loadtxt(data_path + "performance-data.txt")
    best_features = torch.load(model_path[ : model_path.rindex('/')+1] + "best_features.pt")
    forest = torch.load(model_path)

    n_samples = x.shape[0]

    x1 = x[:, best_features]
    y1 = np.argmin(y, axis=1)    # convert to class labels

    predict = []
    for clf in forest:
      y_pred = clf.predict(x1) # pair-wise classifier
      predict.append(y_pred)
    
    # collate instance-wise predictions from all pair-wise classifiers
    result = []
    for i in range(n_samples):
        guess = []
        for p in predict:
            guess.append(p[i])
        result.append(np.argmax(np.bincount(guess))) # majority voting of ensemble. 

    correct = 0
    correct += (result == y1).sum()
    accuracy = correct / n_samples

    print(f"\nFinal results: accuracy: {accuracy:4.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained AS model on a test set")
    parser.add_argument("--model", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    
    args = parser.parse_args()

    print(f"\nLoading trained model {args.model} and evaluating it on {args.data}")
    
    # load the given model, make predictions on the given dataset and evaluate the model's performance. Your evaluation should report four evaluation metrics: avg_loss, accuracy, avg_cost, sbs_vbs_gap (as listed below)
    # you should also calculate the average cost of the SBS and the VBS
    avg_loss = np.inf # the average loss value across the given dataset
    accuracy = 0 # classification accuracy 
    avg_cost = np.inf # the average cost of the predicted algorithms on the given dataset
    sbs_vbs_gap = np.inf # the SBS-VBS gap of your model on the given dataset
    sbs_avg_cost = np.inf # the average cost of the SBS on the given dataset 
    vbs_avg_cost = np.inf # the average cost of the VBS on the given dataset
    # YOUR CODE HERE

    if 'part1' in args.model:
        regression(args.data, args.model)
    elif 'part2_basic' in args.model:
        classification(args.data, args.model)
    elif 'part2_advanced' in args.model:
        classification2(args.data, args.model)
    elif 'forest' in args.model:
        random_forest(args.data, args.model)



    # print results
    # print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")


if __name__ == "__main__":
    main()
