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

data_dir = '/cs/home/akhkr1/Documents/synthetic/'

input_files = os.listdir(data_dir+'input/')
input_files.sort()
target_files = os.listdir(data_dir+'target/')
target_files.sort()

folder_name = 'super_res_model'

epochs = 2291 - 1600
batch_size = 5
rnd_max = 2291

total_training_loss = []
file_num = 0

def plot_predicted_depth(prediction,target_file_name,index):

    i_output = prediction.detach().numpy()
    formatted = i_output.astype('uint8')
    i_depth = Image.fromarray(formatted)
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
    
    
class SuperResModel(nn.Module):
    def __init__(self):
        super(SuperResModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(224*224, 1024*2),
            nn.Softplus(),
            nn.Linear(1024*2, 480*480),
            nn.Softplus()
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = selffc_layers(x)
        x = x.view(x.size(0), 480, 480)
        return x
    
    
start = time.time()

super_res_model = SuperResModel()
#super_res_model.load_state_dict(torch.load(f'/cs/home/akhkr1/Documents/{folder_name}/syn_model.pt'))
super_res_model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = CustomLoss()

for epoch in range(epochs):
    i = random.randint(1601,rnd_max)
    for batch in range(batch_size):
        
        predicted_depth = Image.open(data_dir+'input/'+input_files[i])
        gt_depth_image = Image.open(data_dir+'target/'+target_files[i]).convert('L').resize((480,480))
        
        super_res_pred_depth = super_res_model(predicted_depth).squeeze(0)
        
        gt_np = np.array(gt_depth_image).astype(np.float32)
        gt = torch.tensor(gt_np)
        
        # Compute the loss
        loss = criterion(super_res_pred_depth, gt)
        
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
    if epoch%100 == 0:        
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
    model_file.write(f'Super resolution model. Linear, softplus, 2 layers, 1024_2_2, with Scaled scale invariant exponential loss. Input 224*224 predicted depth, output 480*480 upscaled depth image, lr=1e-4, run {batch_size} times for {epochs} images')