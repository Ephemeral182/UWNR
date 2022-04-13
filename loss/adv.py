
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class DiscriminatorFunieGAN(nn.Module):
    """ A 4-layer Markovian discriminator as described in the paper
    """
    def __init__(self, in_channels=3):
        super(DiscriminatorFunieGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.Conv2d(256, 1, 3, padding=1, bias=False)
        )
        self.Liner = nn.Sequential(
            nn.Linear(196,512),
            nn.LeakyReLU(),
            nn.Linear(512,1)
        )


    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        img_input = img
        y =  self.model(img_input)
        B,C,H,W = y.size()
        y = y.view(B,H*W)
        y =  self.Liner(y)
        y = nn.functional.relu(nn.functional.tanh(y))
        return y


class AdversialLoss(torch.nn.Module):
    def __init__(self):
        super(AdversialLoss,self).__init__()
        model = DiscriminatorFunieGAN().cuda().eval()
        model.load_state_dict(torch.load(r'/mnt/data/yt/Documents/Zero-Reference-Underwater-Image-Enhancedment/Discrininator-epoch15-psnr0.01.pth'))
        self.Net = model

    def forward(self,x):
        B,C,W,H = x.size()
        y = self.Net(x) 
        y = torch.mean(y)
        return y

# loss = AdversialLoss()
# input_ = torch.ones((10,3,224,224)).cuda()
# out = loss(input_)