import torch.nn as nn

from torchvision import models

class VisionNavNet(nn.Module):
    '''
    The model used for vision based navigation.
    -------------------------------------------
    Inputs:  img - an image of size [240, 320]
    Outputs: heading_distance - [robot heading, distance ratio]
    '''

    def __init__(self):

        super().__init__()

        resnet18 = models.resnet18()

        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        self.conv     = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.fc       = nn.Linear(64*4*5, 64)
        self.dropout  = nn.Dropout(p=0.5)
        self.reg      = nn.Linear(64, 2)

    def forward(self, img):

        x = self.features(img)
        x = self.conv(x)
        x = x.view(-1, 64*4*5)
        x = self.fc(x)
        x = self.dropout(x)

        heading_distance = self.reg(x)

        return heading_distance