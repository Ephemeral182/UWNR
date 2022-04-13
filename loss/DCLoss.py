import torch
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.autograd import Variable
from torchvision import transforms

def DCLoss(img, patch_size):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
    dc = maxpool(0-img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()) 
     
    loss = L1Loss(size_average=True)(-dc, target)
    return loss

class DCLoss_Two(nn.Module):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    def __init__(self,patch_size):
        super(DCLoss_Two,self).__init__()
        self.patch_size = patch_size
    def forward(self,img1,img2):
        maxpool = nn.MaxPool3d((2, self.patch_size, self.patch_size), stride=1, padding=(0, self.patch_size//2, self.patch_size//2))
        dc1 = maxpool(0-img1[:, None, :, :, :])
        dc2 = maxpool(0-img2[:, None, :, :, :])
        #target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()) 
        
        loss = L1Loss(size_average=True)(-dc1,-dc2)
        return loss

# a = torch.ones([3,3,20,20])
# b = torch.rand([3,3,20,20])
# d = DCLoss_Two(15)
# print(d(a,b))