from math import exp
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import MinMaxScaler
import warnings
import random
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings('ignore')

torch.device('cuda')

dino_s = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

data_dir = '/cs/home/akhkr1/Documents/synthetic/'

input_files = os.listdir(data_dir+'input/')
input_files.sort()
target_files = os.listdir(data_dir+'target/')
target_files.sort()

folder_name = 'gs_64_64'

epochs = 100
batch_size = 5
rnd_max = 1600

total_training_loss = []
file_num = 0


transform = T.Compose([
        #T.Resize(224, 224),
        #T.CenterCrop(224),
        T.ToTensor()
    ])

'''
transform_1 = T.Compose([
        T.Resize(384),
        T.CenterCrop(384),
        T.ToTensor(),
    ])

transform_depth = T.Compose([
    #T.Resize((64, 64))
    #T.CenterCrop(64)
])
'''

scaler = MinMaxScaler()

def scale_the_input():
    for i in range(rnd_max):
        x = Image.open(data_dir+'input/'+input_files[i]).convert('L').resize((224,224))
        x = transform(x)
        x = torch.concat((x,x,x),dim=0).unsqueeze(0)
        with torch.no_grad():
            dino_features = dino_s.forward_features(x)
            patch_tokens = dino_features['x_norm_patchtokens']
            cls_tokens = dino_features['x_norm_clstoken']
        
        #print(patch_tokens.shape)
        #print(cls_tokens.shape)
        
        #concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens, transform_1(image.convert("L"))),dim=1)
        concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens),dim=1).squeeze(0)
        
        #print(concat.shape)
        if i == 0:
            concat_np = concat.detach().numpy()
            #patch_np = patch_tokens.squeeze(0).detach().numpy()
        else:
            concat_np = np.vstack((concat_np,concat.detach().numpy()))
            #patch_np = np.vstack((patch_np, patch_tokens.squeeze(0).detach().numpy()))
            
    scaler.fit(concat_np)
    #scaler.fit(patch_np)

def plot_predicted_depth(prediction,target_file_name,index):

    i_output = prediction.detach().numpy()
    formatted = i_output.astype('uint8')
    i_depth = Image.fromarray(formatted)
    #print(i_depth.size)
    i_depth.save(f'/cs/home/akhkr1/Documents/{folder_name}/depth_{index}_{target_file_name}')
    
    

def ssim_loss(depth_pred, depth_gt):
    depth_gt_np = depth_gt.detach().numpy()
    depth_pred_np = depth_pred.detach().numpy()
    ssim_loss = (1.0 - ssim(depth_gt_np, depth_pred_np, data_range = depth_pred_np.max() - depth_pred_np.min()))
    return ssim_loss

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mseLoss = torch.nn.MSELoss()

    def forward(self, depth_pred, depth_gt):
        mse_loss = self.mseLoss(depth_pred, depth_gt)
        ssim_loss_value = torch.tensor(ssim_loss(depth_pred, depth_gt), dtype=torch.float32)

        total_loss = mse_loss * exp(ssim_loss_value)

        return total_loss

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(257*384, 1024*3),
            nn.ReLU(),
            nn.Linear(1024*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64*64),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)  # Passing through the fully connected layers
        x = x.view(x.size(0), 64, 64)
        return x  

scale_the_input()
    
start = time.time()

model = DepthEstimationModel()


model.train()
#model.load_state_dict(torch.load(f'/cs/home/akhkr1/Documents/{folder_name}/syn_model.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#criterion = torch.nn.MSELoss()
criterion = CustomLoss()


for epoch in range(epochs):
    i = random.randint(0,rnd_max)
    for batch in range(batch_size):
        
        rgb_image = Image.open(data_dir+'input/'+input_files[i]).convert('L').resize((224,224))
        gt_depth_image = Image.open(data_dir+'target/'+target_files[i]).convert('L').resize((64,64))
        
        rgb = transform(rgb_image)
        
        rgb = torch.concat((rgb,rgb,rgb),dim=0).unsqueeze(0)
        with torch.no_grad():
          dino_features = dino_s.forward_features(rgb)
        
        patch_tokens = dino_features['x_norm_patchtokens'] #[1, 256, 384]
        cls_tokens = dino_features['x_norm_clstoken'] #[1, 384]
        concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens),dim=1).squeeze(0)
        
        concat_norm = scaler.transform(concat.detach().numpy())
                
        concat_norm = torch.tensor(concat_norm,dtype=torch.float32).unsqueeze(0)
        
        #concat_norm = torch.cat((concat_norm, transform_1(rgb_image.convert("L"))),dim=1).unsqueeze(0)
        
        
        #features = scaler.transform(patch_tokens.squeeze(0).detach().numpy())
        
        #features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        predicted_depth = model(concat_norm).squeeze(0)
        #predicted_depth = model(features).squeeze(0)
        

        #gt_depth = transform_depth(gt_depth_image)
        
        gt_np = np.array(gt_depth_image).astype(np.float32)
        gt = torch.tensor(gt_np)
        
        # Compute the loss
        loss = criterion(predicted_depth, gt)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        total_training_loss.append(train_loss)
        #print(train_loss)

        file_num += 1
        with open('/cs/home/akhkr1/Documents/logs.txt','a') as f:
            f.write(str(file_num))
            f.write(f' file name: input{i}.jpg train loss: {train_loss}\n')
    if epoch%10 == 0:
        with torch.no_grad():
            plot_predicted_depth(predicted_depth,f'target{i}.png',epoch)

end = time.time()
total_time = end - start
with open(f'/cs/home/akhkr1/Documents/{folder_name}/syn_model.txt','a') as time_file:
    time_file.write(f'\nTime taken to train the model: {total_time}\n')

torch.save(model.state_dict(),f'/cs/home/akhkr1/Documents/{folder_name}/syn_model.pt')
loss_np = np.array(total_training_loss)
np.savetxt(f'/cs/home/akhkr1/Documents/{folder_name}/train_loss_sun.txt', loss_np, delimiter=',')

fig, ax = plt.subplots()
# Plot the list data
ax.plot(total_training_loss)

# Add labels and title
ax.set_title('Loss')
# Save the plot as an image
fig.savefig(f'/cs/home/akhkr1/Documents/{folder_name}/loss_plot_sun.png')
# Close the plot
plt.close(fig)

with open(f'/cs/home/akhkr1/Documents/{folder_name}/syn_model.txt', 'a') as model_file:
    model_file.write(f'Without cropping. Grayscale input to dino. Linear, softplus, 3 layers, 1024*3, 1024, with Scaled scale invariant exponential loss. Input 257*384, output 64*64, lr=1e-4, batch size is {batch_size} for {epochs} epochs')
