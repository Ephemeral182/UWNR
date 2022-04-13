import random

from torch.autograd import Variable
import torch
# from visdom import Visdom
import torchvision.transforms as transforms
import numpy as np
from skimage.filters import threshold_otsu


to_pil = transforms.ToPILImage()
to_gray = transforms.Grayscale(num_output_channels=1)

class QueueMask():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__()-1]



def mask_generator(shadow, shadow_free):
	im_f = to_gray(to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5).cpu()))
	im_s = to_gray(to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5).cpu()))

	diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
	L = threshold_otsu(diff)
	mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0).cuda() #-1.0:non-shadow, 1.0:shadow
	mask.requires_grad = False

	return mask



def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)



class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv2d') != -1:
        if classname.find('DeformableConv2d') != -1:
            pass
        else:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



