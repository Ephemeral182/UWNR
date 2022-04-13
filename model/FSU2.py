import torch
import torch.nn as nn
from torch.nn.modules.activation import ELU, LeakyReLU, Sigmoid
from torch.nn.modules.upsampling import Upsample
#from model.dcn import DeformableConv2d
#from DCNv2.dcn_v2 import DCN
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,conv=nn.Conv2d,act=nn.ELU):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            conv(in_ch, out_ch, 3, padding=1),
            act(inplace=True),
            conv(out_ch, out_ch, 3, padding=1),
            act(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x) + x
        return x
    
class downsample(nn.Module):
    def __init__(self, in_ch, out_ch,conv=nn.Conv2d,act=nn.ELU):
        super(downsample, self).__init__()
        self.mpconv = nn.Sequential(
            conv(in_ch, out_ch,kernel_size=3, stride=2,padding=3//2), 
            act(inplace=True),
            BasicBlock(out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
    
class upsample(nn.Module):
    def __init__(self,in_ch,conv = nn.Conv2d,act=nn.ELU):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            conv(in_ch, 2*in_ch, kernel_size=3,stride=1,padding=3//2),
            nn.ELU(),
            nn.PixelShuffle(2),
        )
    def forward(self, x):
        y = self.up(x)
        return y
    
class SpatialAttention(nn.Module):
    def __init__(self,chns,factor):
        super(SpatialAttention,self).__init__()
        self.spatial_pool = nn.Sequential(
            nn.Conv2d(chns,chns//factor,1,1,0),
            nn.LeakyReLU(),
            nn.Conv2d(chns//factor,1,1,1,0),
            nn.Sigmoid()
        )
    def forward(self,x):
        #X @ B,C,H,W
        #map @B,1,H,W
        spatial_map = self.spatial_pool(x)
        return x*spatial_map


class ChannelAttention(nn.Module):
    def __init__(self,chns,factor):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_map = nn.Sequential(
            nn.Conv2d(chns,chns//factor,1,1,0),
            nn.LeakyReLU(),
            nn.Conv2d(chns//factor,chns,1,1,0),
            nn.Sigmoid()
        )
    def forward(self,x):
        avg_pool = self.avg_pool(x)
        map = self.channel_map(avg_pool)
        return x*map

# x = torch.rand([12,64,256,256])
# y = SpatialAttention(64,16)
# x = y(x) 
class BasicBlock(nn.Module):
    def __init__(self,chns):
        super(BasicBlock,self).__init__()
        self.conk3 = nn.Conv2d(chns,chns,3,1,3//2)
        self.conk1 = nn.Conv2d(chns,chns,1,1,1//2)
        # self.model = nn.Sequential(
        # nn.Conv2d(chns,chns,3,1,3//2),
        # nn.LeakyReLU(inplace=True),
        # SpatialAttention(chns,4),
        # ChannelAttention(chns,4),
        # )
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.SA = SpatialAttention(chns,4)
        self.CA = ChannelAttention(chns,4)
        self.norm = nn.InstanceNorm2d(chns//2, affine=True)
    def forward(self,x):
        residual = x
        y = self.conk1(x)+ self.conk3(x) + residual
        
        
        #x = self.model(x)+residual
        output = self.leakyrelu(y)
        output = self.CA(self.SA(output))+residual
        #output = self.CA((output))+residual
        #output = (self.SA(output))+residual
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.in_conv_down1 = downsample(7,64)
        self.down2 = downsample(64,128)
        self.down3 = downsample(128,256)
        self.down4 = downsample(256,512)
       
        self.up1 = upsample(512)
        self.up2 = upsample(256)  #in@128 out@64
        self.up3 = upsample(128)  #in@64 out@32
        self.up4 = upsample(64) #in@16
        
        self.out = nn.Sequential(
            nn.Conv2d(32,3,3,1,1),
            nn.Tanh(),
        )
        
        #self.deConv1 = DeformableConv2d(512,512,3,1,1)

    def forward(self,x):
        C,B,H,W = x.shape
        residual = x[:,0:3,:,:]
        x2 = self.in_conv_down1(x)
        x4 = self.down2(x2)
        x8 = self.down3(x4)
        x16 = self.down4(x8)
        
        #y = self.deConv1(x16)
        y = x16
        y8 = self.up1(y)+x8
        y4 = self.up2(y8)+x4
        y2 = self.up3(y4)+x2
        y = self.up4(y2)
        
        out = self.out(y) 
        return out


