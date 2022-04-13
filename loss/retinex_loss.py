import torch.nn as nn
import numpy as np
from loss.perceptual import *
from PIL import Image
import torch
import numpy as np
import math


r = 0
s = [15,60,90]
class MyGaussianBlur(torch.nn.Module):
    #初始化
    def __init__(self, radius=1, sigema=1.5):
        super(MyGaussianBlur,self).__init__()
        self.radius=radius
        self.sigema=sigema
    #高斯的计算公式
    def calc(self,x,y):
        res1=1/(2*math.pi*self.sigema*self.sigema)
        res2=math.exp(-(x*x+y*y)/(2*self.sigema*self.sigema))
        return res1*res2
 
    #滤波模板
    def template(self):
        sideLength=self.radius*2+1
        result=np.zeros((sideLength, sideLength))
        for i in range(0, sideLength):
            for j in range(0,sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all= result.sum()
        return result/all
    #滤波函数
    def filter(self, image, template):
        kernel = torch.FloatTensor(template).cuda()
        kernel2 = kernel.expand(3, 1, 2*r+1, 2*r+1)
        weight = torch.nn.Parameter(data=kernel2, requires_grad=False)
        new_pic2 = torch.nn.functional.conv2d(image, weight, padding=r, groups=3)
        return new_pic2

# print(loss.item())
def MutiScaleLuminanceEstimation(img):

    guas_15 = MyGaussianBlur(radius=r, sigema=15).cuda()
    temp_15 = guas_15.template()
        
    guas_60 = MyGaussianBlur(radius=r, sigema=60).cuda()
    temp_60 = guas_60.template()

    guas_90 = MyGaussianBlur(radius=r, sigema=90).cuda()
    temp_90 = guas_90.template()
    x_15 = guas_15.filter(img, temp_15)
    x_60 = guas_60.filter(img, temp_60)
    x_90 = guas_90.filter(img, temp_90)
    img = (x_15+x_60+x_90)/3

    return img
class Retinex_loss1(nn.Module):
    def __init__(self):
        super(Retinex_loss1,self).__init__()
        
        self.L1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
    def forward(self,y,x):

        batch_size,h_x,w_x = x.size()[0],x.size()[2],x.size()[3]
        x = MutiScaleLuminanceEstimation(x)
        y = MutiScaleLuminanceEstimation(y)

        retinex_loss_L1 = self.L1(x,y)
        
        return retinex_loss_L1
# a = torch.rand([3,3,256,256],requires_grad=True).cuda()
# b = torch.rand([3,3,256,256],requires_grad=True).cuda()
# losss = Retinex_loss1().cuda()
# loss = losss(a,b)
# loss.backward()
# print(loss)