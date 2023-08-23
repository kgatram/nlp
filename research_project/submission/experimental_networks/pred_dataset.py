import torch
import torchvision.transforms as T
import numpy as np
import torch.nn as nn
from PIL import Image
import os
import time
import warnings
import random
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings('ignore')

torch.device('cuda')

dino_s = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

data_dir = '/cs/home/akhkr1/Documents/synthetic/'

input_files = os.listdir(data_dir+'input/')
input_files.sort()

folder_name = 'pred_dataset'

file_size = 2291

total_training_loss = []
file_num = 0

transform = T.Compose([
        #T.Resize(224),
        #T.CenterCrop(224),
        T.ToTensor()
    ])

scaler = MinMaxScaler()

def scale_the_input():
    for i in range(rnd_max):
        x = Image.open(data_dir+'input/'+input_files[i])
        x = x.resize((224,224))
        image_dino = transform(x).unsqueeze(0)
        with torch.no_grad():
            dino_features = dino_s.forward_features(image_dino)
            patch_tokens = dino_features['x_norm_patchtokens']
            cls_tokens = dino_features['x_norm_clstoken']
            
        concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens),dim=1).squeeze(0)
        
        if i == 0:
            concat_np = concat.detach().numpy()
        else:
            concat_np = np.vstack((concat_np,concat.detach().numpy()))
            
    scaler.fit(concat_np)

def plot_predicted_depth(prediction,target_file_name):

    i_output = prediction.detach().numpy()
    formatted = i_output.astype('uint8')
    i_depth = Image.fromarray(formatted)
    i_depth.save(f'/cs/home/akhkr1/Documents/{folder_name}/pred_depth_{target_file_name}')
    
class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(257*384, 1024*2),
            nn.Softplus(),
            nn.Linear(1024*2, 1024*2),
            nn.Softplus(),
            nn.Linear(1024*2, 224*224),
            nn.Softplus()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)  # Passing through the fully connected layers
        x = x.view(x.size(0), 224, 224)
        return x  
    
scale_the_input()

depth_model = DepthEstimationModel()
depth_model.load_state_dict(torch.load('/cs/home/akhkr1/Documents/resize_exp_loss/syn_model.pt'))
depth_model.eval()

for i in range(file_size):
    rgb_image = Image.open(data_dir+'input/'+input_files[i]).resize((224,224))        
    
    rgb = transform(rgb_image).unsqueeze(0)
    with torch.no_grad():
        dino_features = dino_s.forward_features(rgb)
    
    patch_tokens = dino_features['x_norm_patchtokens'] #[1, 256, 384]
    cls_tokens = dino_features['x_norm_clstoken'] #[1, 384]
    concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens),dim=1).squeeze(0)
    
    concat_norm = scaler.transform(concat.detach().numpy())  
    concat_norm = torch.tensor(concat_norm,dtype=torch.float32).unsqueeze(0)

    predicted_depth = depth_model(concat_norm).squeeze(0)
       
    plot_predicted_depth(predicted_depth,f'{i}.png')