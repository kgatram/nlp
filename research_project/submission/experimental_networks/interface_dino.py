import cv2
import torch
import torch.nn as nn
import numpy
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F

dino_s = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

transform_1 = T.Compose([
        T.Resize(384),
        T.CenterCrop(384),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])



class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(641*384, 1024*5),
            #nn.Linear(257*384, 1024*5),
            nn.Softplus(),
            nn.Linear(1024*5, 1024*3),
            nn.Softplus(),
            nn.Linear(1024*3, 1024*5),
            nn.Softplus(),
            nn.Linear(1024*5, 224*224),
            nn.Softplus()
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flattening the extracted features (256x384)
        x = self.fc_layers(x)  # Passing through the fully connected layers
        x = x.view(x.size(0),224,224)  # Reshaping the output to (640, 480)
        return x

model = DepthEstimationModel()
#model.load_state_dict(torch.load('/cs/home/akhkr1/cuda-ubuntu/depth_transformer/linear_4_1024_2048_softplus/dino_depth_softplus.pt'))
model.load_state_dict(torch.load('/home/akhkr1/Downloads/image_1024_5_3_5/dino_depth_softplus.pt'))
model.eval()

timestamp = 0

capture = cv2.VideoCapture(0)
#reading the camera feed
while True:
    ret, image = capture.read()
    #image = Image.open('/cs/home/akhkr1/cuda-ubuntu/tum_dataset/rgb/1305031458.927621.png')
    #image = Image.open('r.ppm')
    image = Image.fromarray(image)
    image_dino = transform(image).unsqueeze(0)
    with torch.no_grad():
        dino_features = dino_s.forward_features(image_dino)
    patch_tokens = dino_features['x_norm_patchtokens']
    cls_tokens = dino_features['x_norm_clstoken']
    concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens, transform_1(image.convert("L"))),dim=1)
    #concat = torch.cat((cls_tokens.unsqueeze(0), patch_tokens),dim=1)
    # Forward pass
    predicted_depth = model(concat).squeeze(0)

    # interpolate to original size
    i_prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(0).unsqueeze(0),
                        size=image.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                  ).squeeze()
    i_output = i_prediction.detach().cpu().numpy().astype('uint8')
    #formatted = (i_output * 255 / np.max(i_output)).astype('uint8')
    i_depth = Image.fromarray(i_output)

    timestamp += 1


    i_depth.save(f'/home/akhkr1/Downloads/orb_depth/depth_{timestamp}.png')
    image.save(f'/home/akhkr1/Downloads/orb_rgb/rgb_{timestamp}.png')