import torch
import torch.nn as nn
import torch.nn.functional as F

from .SVAE import SVAE
from .VisionNavNet import VisionNavNet

class ImageModel(nn.Module):
    '''
    Image branch of the model processing the image.
    -------------------------------------------
    Inputs:  img - an image of size [240, 320]
    Outputs: img_features - feature maps of the image
    Note:    ImageModel has a downsampling rate of 64.
             The output has a dimension of 4*5.
    '''

    def __init__(self, freeze_features, pretrained_file):

        super().__init__()

        # load pretrained weights
        vision_nav_net = VisionNavNet()
        checkpoint = torch.load(pretrained_file)
        vision_nav_net.load_state_dict(checkpoint['state_dict'])

        if freeze_features:
            for param in vision_nav_net.parameters():
                param.requires_grad = False

        self.features = nn.Sequential(*list(vision_nav_net.children())[:-3])

        # add extra convolutional layers
        self.conv_L_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_A_1 = nn.ReLU()

    def forward(self, img):

        img_features = self.features(img)
        img_features = self.conv_A_1(self.conv_L_1(img_features))

        return img_features

class TrajModel(nn.Module):
    '''
    Trajectory branch of the model processing the predicted trajectory.
    ------------------------------------------------------
    Inputs:  traj - a projected future trajectory
    Outputs: traj_features - feature maps of the future trajectory
    Note:    The output has a dimension of 4*5
    '''

    def __init__(self):

        super().__init__()
        
        self.conv_L_1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv_A_1 = nn.ReLU()

        self.pool_1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        self.conv_L_2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv_A_2 = nn.ReLU()

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_L_3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv_A_3 = nn.ReLU()

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, traj):

        traj_features = self.conv_A_1(self.conv_L_1(traj))
        traj_features = self.pool_1(traj_features)
        traj_features = self.conv_A_2(self.conv_L_2(traj_features))
        traj_features = self.pool_2(traj_features)
        traj_features = self.conv_A_3(self.conv_L_3(traj_features))
        traj_features = self.pool_3(traj_features)

        return traj_features

class PAAD(nn.Module):
    '''
    The complete model for predictive anomaly detection.
    ----------------------------------------------------
    Inputs:  img - an image of size [240, 320]
             traj - a projected future trajectory
             lidar_scan - a vector of range readings of size 1081
    Outputs: pred_inv_score - a sequence of future intervention probabilities
    '''

    def __init__(self, device, freeze_features, pretrained_file, horizon):

        super().__init__()

        self.image_model = ImageModel(freeze_features, pretrained_file)
        self.traj_model  = TrajModel()
        self.lidar_model = SVAE(device)

        self.img_L_1 = nn.Linear(640, 64)
        self.img_A_1 = nn.ReLU()

        self.traj_L_1 = nn.Linear(640, 64)
        self.traj_A_1 = nn.ReLU()

        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8) # 8

        self.fc_L_1 = nn.Linear(64+64+64, 128) # (64+64+64, 128)
        self.fc_A_1 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_2 = nn.Linear(128, horizon) # 128
        self.fc_A_2 = nn.Sigmoid()

    def forward(self, img, traj, lidar_scan):

        img_features  = self.image_model(img)
        traj_features = self.traj_model(traj)
        recon_lidar, means, log_var = self.lidar_model(lidar_scan)

        img_features  = torch.flatten(img_features, 1)
        traj_features = torch.flatten(traj_features, 1)

        img_features   = self.img_A_1(self.img_L_1(img_features))
        traj_features  = self.traj_A_1(self.traj_L_1(traj_features))
        lidar_features = torch.cat((means, log_var), dim=-1)

        img_features   = img_features.unsqueeze(0)
        lidar_features = lidar_features.unsqueeze(0)
        cat_features   = torch.cat((img_features, lidar_features), dim=0)

        cat_features, cat_features_weights = self.mha(cat_features, cat_features, cat_features)

        cat_features = cat_features + torch.cat((img_features, lidar_features), dim=0)

        cat_features = cat_features.permute(1, 0, 2)
        cat_features = torch.flatten(cat_features, 1)

        # concatenate feature maps
        cat_features = torch.cat((cat_features, traj_features), dim=-1)

        # fc layer
        cat_features   = self.fc_A_1(self.fc_L_1(cat_features))
        cat_features   = self.dropout(cat_features)
        pred_inv_score = self.fc_A_2(self.fc_L_2(cat_features))

        return recon_lidar, means, log_var, pred_inv_score