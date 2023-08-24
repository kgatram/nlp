from math import exp
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

torch.device('cuda')
'''

files = [
    '1305031453.png',
'1305031454.png',
'1305031455.png',
'1305031456.png',
'1305031457.png',
'1305031458.png',
'1305031459.png',
'1305031460.png',
'1305031461.png',
'1305031462.png',
'1305031463.png',
'1305031464.png',
'1305031465.png'
]

files = [
    'NYU0001',
    'NYU0002',
    'NYU0003']
'''

#rgb_files,depth_files = np.loadtxt('/cs/home/akhkr1/cuda-ubuntu/tum_dataset/tum_synced.txt',dtype='str',delimiter=' ',usecols=(1,3),unpack=True)

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

folder_name = 'no_image_1024_5_3_5'
transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor()
        #T.Normalize(mean=[0.5], std=[0.5]),
    ])

transform_1 = T.Compose([
        T.Resize(384),
        T.CenterCrop(384),
        T.ToTensor(),
        #T.Normalize(mean=[0.5], std=[0.5]),
    ])

transform_depth = T.Compose([
    T.Resize((224)),
    T.CenterCrop(224)
])

def plot_predicted_depth(prediction,target_file_name,index):
    input_file_name = input_file_name.replace('/','_')
    target_file_name = target_file_name.replace('/','_')
    '''
    # interpolate to original size
    i_prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(0).unsqueeze(0),
                    size=input_image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
              ).squeeze()
    '''
    i_output = prediction.detach().numpy()
    formatted = i_output.astype('uint8')
    i_depth = Image.fromarray(formatted)
    #ground_truth = ground_truth.astype('uint8')
    #exp_depth = Image.fromarray(ground_truth)
    #print(i_depth.size)
    i_depth.save(f'/home/akhkr1/Downloads/{folder_name}/depth_{index}_{target_file_name}')
    #exp_depth.save(f'/home/akhkr1/Downloads/{folder_name}/gt_at_{target_file_name}')
    #input_image.save(f'/home/akhkr1/Downloads/{folder_name}/input_at_{input_file_name}')


class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(257*384, 1024*5),
            #nn.Linear(641*384, 1024*5),
            nn.Softplus(),
            nn.Linear(1024*5, 1024*5),
            nn.Softplus(),
            #nn.Linear(1024*3, 1024*5),
            #nn.Softplus(),
            nn.Linear(1024*5, 224*224),
            nn.Softplus()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        #print(x.shape)
        x = self.fc_layers(x)  # Passing through the fully connected layers
        x = x.view(x.size(0),224,224) 
        return x

model = DepthEstimationModel()
model.train()
#model.load_state_dict(torch.load('/home/akhkr1/Downloads/no_image_1024_5_3_5/dino_syn.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()


#scale_factor = 5000

epochs = 3
#data_dir = '/cs/home/akhkr1/cuda-ubuntu/tum_dataset/'
#data_dir = '/home/akhkr1/Downloads/'
#data_dir='/cs/home/akhkr1/Documents/kv1/NYUdata/'
data_dir = '/cs/home/akhkr1/Documents/synthetic/'

total_training_loss = []
file_num = 0
input_files = os.listdir(data_dir+'input/')
input_files.sort()
target_files = os.listdir(data_dir+'target/')
target_files.sort()


for epoch in range(epochs):
    for i in range(1391):
    #for i in range(len(files)):
        #input_image_name = rgb_files[i]
        #target_image_name = depth_files[i]
        #file_name = 'NYU' + str(i).zfill(4)

        #input_image_name = files[i]
        #target_image_name = files[i]
        
        #rgb_image = Image.open(data_dir+input_image_name)
        #gt_depth_image = Image.open(data_dir+target_image_name)
        
        #rgb_image = Image.open(data_dir+file_name+'/fullres/'+file_name+'.jpg')
        #gt_depth_image = Image.open(data_dir+file_name+'/fullres/'+file_name+'.png')

        #rgb_image = Image.open(data_dir+'sun_rgb/'+input_image_name)
        #gt_depth_image = Image.open(data_dir+'sun_depth/'+target_image_name)
        
        rgb_image = Image.open(data_dir+'input/'+input_files[i])
        gt_depth_image = Image.open(data_dir+'target/'+target_files[i]).convert('L')
        
        rgb = transform(rgb_image).unsqueeze(0)
        with torch.no_grad():
          dino_features = dino.forward_features(rgb)
        
        patch_tokens = dino_features['x_norm_patchtokens']
        cls_tokens = dino_features['x_norm_clstoken']
        #print('patch_tokens:', patch_tokens.shape)
        #print('cls_tokens.unsqueeze:',cls_tokens.unsqueeze(0).shape)
        #print('image: ',transform_1(rgb_image.convert("L")).shape)

        #concat = torch.cat((cls_tokens.repeat(3,1,1), patch_tokens.repeat(3,1,1), transform_1(rgb_image)),dim=1)
        #concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens, transform_1(rgb_image.convert("L"))),dim=1)
        concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens),dim=1)
        #print(concat.shape)
        predicted_depth = model(concat).squeeze(0)
        

        gt_depth = transform_depth(gt_depth_image)
        
        gt_np = np.array(gt_depth).astype(np.float32)
        gt = torch.tensor(gt_np)
        '''
        # Convert the ground truth depth to integers
        gt_depth = (gt_depth / scale_factor)
        gt = gt_depth.detach().cpu().numpy()
        gt = (gt * 255 / np.max(gt))
        gt = torch.tensor(gt)
        '''
        #print(gt.shape)
        with torch.no_grad():
            if i%500 == 0:
                plot_predicted_depth(rgb_image,predicted_depth, gt_np,f'input{i}.jpg',f'target{i}.png',epoch)

        
        # Compute the loss
        #loss = criterion(predicted_depth, gt_depth.repeat(3,1,1))
        loss = criterion(predicted_depth, gt)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        total_training_loss.append(train_loss)
        #print(train_loss)

        file_num += 1
        with open('/home/akhkr1/Downloads/logs.txt','a') as f:
            f.write(str(file_num))
            f.write(f' file name: input{i}.jpg train loss: {train_loss}\n') 

torch.save(model.state_dict(),f'/home/akhkr1/Downloads/{folder_name}/dino_syn.pt')
loss_np = np.array(total_training_loss)
np.savetxt(f'/home/akhkr1/Downloads/{folder_name}/train_loss_sun.txt', loss_np, delimiter=',')

fig, ax = plt.subplots()
# Plot the list data
ax.plot(total_training_loss)

# Add labels and title
ax.set_title('Loss')
# Save the plot as an image
fig.savefig(f'/home/akhkr1/Downloads/{folder_name}/loss_plot_sun.png')
# Close the plot
plt.close(fig)
