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
import pickle
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
# enabling cuda for GPU
torch.device('cuda')

start = time.time()

# loading the visual transformer
dino_s = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


folder_name = '500_epochs_downsize'

epochs = 100

epoch_training_loss = []

transform_input = T.Compose([
     T.ToTensor()
])

# loading the scaler pickle file
scaler = MinMaxScaler()
try:
    with open('/cs/home/akhkr1/Documents/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except IOError:
    print('error')

# plots the images from a tensor
def plot_predicted_depth(prediction, index):
    i_output = prediction.detach().numpy()
    formatted = i_output.astype('uint8')
    i_depth = Image.fromarray(formatted)
    i_depth.save(f'/cs/home/akhkr1/Documents/{folder_name}/depth_at_epoch_{index}.png')

# defining the Scaled scale invariant loss function
def ssim_loss(depth_pred, depth_gt):
    depth_gt_np = depth_gt.detach().numpy()
    depth_pred_np = depth_pred.detach().numpy()
    ssim_loss = (1.0 - ssim(depth_gt_np, depth_pred_np, data_range=depth_pred_np.max() - depth_pred_np.min()))
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

# defining the feed-forward network
class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(257 * 384, 1024 * 2),
            nn.Softplus(),
            nn.Linear(1024 * 2, 1024 * 2),
            nn.Softplus(),
            nn.Linear(1024 * 2, 32 * 32),
            nn.Softplus()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)  # Passing through the fully connected layers
        x = x.view(x.size(0), 32, 32)
        return x

# defining the custom data loader
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_dir = os.path.join(self.root_dir, 'input/')
        self.target_dir = os.path.join(self.root_dir, 'target/')
        self.input_filenames = os.listdir(self.input_dir)
        self.target_filenames = os.listdir(self.target_dir)

        # Ensure the input and target filenames are in sync
        self.input_filenames.sort()
        self.target_filenames.sort()

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_filenames[idx])
        target_image_path = os.path.join(self.target_dir, self.target_filenames[idx])

        input_to_dino = Image.open(input_image_path).resize((224,224))
        target_image = Image.open(target_image_path).convert('L').resize((32,32))

        input_to_dino = transform_input(input_to_dino).unsqueeze(0)
        with torch.no_grad():
            dino_features = dino_s.forward_features(input_to_dino)

        patch_tokens = dino_features['x_norm_patchtokens']  # [1, 256, 384]
        cls_tokens = dino_features['x_norm_clstoken']  # [1, 384]
        concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens), dim=1).squeeze(0)

        concat_norm = scaler.transform(concat.detach().numpy())

        input_features = torch.tensor(concat_norm, dtype=torch.float32).unsqueeze(0)
        target_image = np.array(target_image).astype(np.float32)
        target_tensor = torch.tensor(target_image)

        return input_features, target_tensor


model = DepthEstimationModel()
model.train()
model.load_state_dict(torch.load(f'/cs/home/akhkr1/Documents/400_epochs_downsize/syn_model.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = CustomLoss()


# Define the root directory where your data is stored
root_dir = '/cs/home/akhkr1/Documents/synthetic/'

# Create the custom dataset with paired input and target images
paired_dataset = ImageDataset(root_dir)

# Create DataLoader with batch size and shuffle
batch_size = 32
paired_loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)

# training the network
for epoch in range(epochs):
    train_loss = 0
    predicted_print = 0
    for input_batch, target_batch in paired_loader:
        predicted_batch = model(input_batch).squeeze(0)
        predicted_print = predicted_batch[26]
        # Compute the loss
        loss = criterion(predicted_batch, target_batch)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
    if epoch%20 == 0:
        with torch.no_grad():
            plot_predicted_depth(predicted_print, epoch)

    epoch_training_loss.append(train_loss)

    with open('/cs/home/akhkr1/Documents/500_epochs_downsize_logs.txt', 'a') as f:
        f.write(f'Training loss for epoch {epoch} of batch {batch_size}: {train_loss}\n')

end = time.time()
total_time = end - start
with open(f'/cs/home/akhkr1/Documents/{folder_name}/syn_model.txt', 'a') as time_file:
    time_file.write(f'\nTime taken to train the model: {total_time}\n')
# saving the network weights
torch.save(model.state_dict(), f'/cs/home/akhkr1/Documents/{folder_name}/syn_model.pt')
loss_np = np.array(epoch_training_loss)
np.savetxt(f'/cs/home/akhkr1/Documents/{folder_name}/train_loss_syn.txt', loss_np, delimiter=',')
# plotting the predictions and saving them
fig, ax = plt.subplots()
# Plot the list data
ax.plot(epoch_training_loss)

# Add labels and title
ax.set_title('Train Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
# Save the plot as an image
fig.savefig(f'/cs/home/akhkr1/Documents/{folder_name}/loss_plot.png')
# Close the plot
plt.close(fig)

with open(f'/cs/home/akhkr1/Documents/{folder_name}/syn_model.txt', 'a') as model_file:
    model_file.write(f'Downsized to 32*32. Linear, softplus, 3 layers, 1024_2_2, with Scaled scale invariant exponential loss. Input 257*384, output 32*32, lr=1e-4, run {epochs} times of {batch_size} batch size')