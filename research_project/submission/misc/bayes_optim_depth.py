import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState

rgb_files,depth_files = np.loadtxt('/cs/home/akhkr1/cuda-ubuntu/tum_dataset/tum_synced.txt',dtype='str',delimiter=' ',usecols=(1,3),unpack=True)

dino_s = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

scale_factor = 5000

transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

transform_depth = T.Compose([
    T.Resize((640, 480)),  # Resize the depth image to match the output shape
    T.ToTensor(),
])


def create_model(trial):
    n_layers = trial.suggest_int('n_layers', 4,6)
    layers = []
    in_features = 256*384
    for i in range(n_layers):
        out_features = trial.suggest_int('n_units_l{}'.format(i),1200,2200)
        layers.append(torch.nn.Linear(in_features,out_features))
        layers.append(torch.nn.GELU())
        p = trial.suggest_float('dropout_l{}'.format(i),0.2,0.5)
        layers.append(torch.nn.Dropout(p))
        in_features = out_features
    
    layers.append(torch.nn.Linear(in_features,640*480))
    
    return torch.nn.Sequential(*layers)

#training
def train(model,features,y,optimizer,criterion):
    model.train()
    
    features = features.view(features.size(0),-1)
    y_hat = model(features).squeeze(0)
    y_hat = y_hat.view(y.size())
    
    loss = criterion(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def objective(trial):

    model = create_model(trial)
    lr = trial.suggest_float('lr',1e-5,1e-1,log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    criterion = torch.nn.SmoothL1Loss()  

    epochs = 50
    data_dir = '/cs/home/akhkr1/cuda-ubuntu/tum_dataset/'

    total_training_loss = []

    for epoch in range(epochs):
        for b in range(10):
            i = random.randint(0,1000)
        
            input_image = rgb_files[i]
            target_image = depth_files[i]
        
            rgb = Image.open(data_dir+input_image)
            depth = Image.open(data_dir+target_image)
        
            rgb = transform(rgb).unsqueeze(0)
            
            gt_depth = transform_depth(depth).squeeze(0)
            # Convert the ground truth depth to integers
            gt_depth = (gt_depth / scale_factor)
        
            with torch.no_grad():
                features = dino_s.forward_features(rgb)["x_norm_patchtokens"]
            train_loss = train(model,features,gt_depth,optimizer,criterion)
            total_training_loss.append(train_loss)
            trial.report(train_loss,epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return total_training_loss[-1]
                
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=10)

pruned_trials = study.get_trials(deepcopy=False,states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False,states=[TrialState.COMPLETE])

print("States:")
print('Number of finished trials: ',len(study.trials))
print('Number of pruned trials: ',len(pruned_trials))
print('Number of completed trials: ',len(complete_trials))

print('Best trial: ')
trial = study.best_trial
print('Value: ',trial.value)

print('Params: ')
for key,value in trial.params.items():
  print(' {}: {}'.format(key,value))