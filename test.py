import argparse
import os
import torchvision.transforms as transforms
import torch
import time
import numpy as np
from metrics import *
from model.FSU2 import *
import warnings
from torchvision.utils import save_image,make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader
from myutils.dataloader import SUID_Dataset, UWS_Dataset_Retinex_test
from PIL import Image
from pytorch_fid_test.src.pytorch_fid.fid_score import *
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--test_size',type=int,default=256,help='test size')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--save_path',type=str,default='/mnt/data/csx/Documents/cvpr2022w_underwater/out',help='the path to save generation underwater image')
parser.add_argument('--clean_img_path',type=str,default='/mnt/data/csx/Documents/underwater_generation2022/NYU_test/NYU_GT',help='the path of saving clean image')
parser.add_argument('--depth_img_path',type=str,default='/mnt/data/csx/Documents/underwater_generation2022/NYU_test/NYU_depth',help='the path of saving depth image')
parser.add_argument('--underwater_path',type=str,default='/mnt/data/csx/Documents/dataset/UIEB',help='the path of saving real underwater image')
parser.add_argument('--model_path',type=str,default='/mnt/data/csx/Documents/underwater_generation2022/train_models/l1+vgg+udcp+reti_l1+cs/epoch_render-epoch200-fid221.93.pk',help='the path of UWNR model pth')
parser.add_argument('--fid_gt_path',type=str,default='/mnt/data/csx/Documents/underwater_generation2022/train_models/UIEB_256',help='the path of real underwater image as fid ground truth')
opt = parser.parse_args()
test_lir=os.listdir(opt.clean_img_path)
test_lir.sort()
test_data =[os.path.join(opt.clean_img_path,test_img)  for test_img in test_lir]

test_lir_depth=os.listdir(opt.depth_img_path)
test_lir_depth.sort()
test_depth =[os.path.join(opt.depth_img_path,test_img)  for test_img in test_lir_depth]
UW_test_loader = DataLoader(dataset=UWS_Dataset_Retinex_test(opt.underwater_path,train=False,size=opt.test_size,dcp=False),batch_size=1,shuffle=False,num_workers=0)

if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')
print(opt)


netG_1 = Generator()
if opt.cuda:
    netG_1.to(device)

ssims = []
psnrs = []

#g1ckpt='/mnt/data/csx/Documents/underwater_generation2022/train_models/l1+vgg+udcp+reti_l1+cs/epoch_render-epoch200-fid221.93.pk'
g1ckpt = opt.model_path
ckpt = torch.load(g1ckpt)
from collections import OrderedDict

new_state = OrderedDict()
for k,v in ckpt['G1'].items():
    k = k[7:]
    new_state[k] = v
netG_1.load_state_dict(new_state)

#savepath='/mnt/data/csx/Documents/dataset/LNRUD/LNRUD_water/'
#savepath_gt='/mnt/data/csx/Documents/dataset/LNRUD/LNRUD_gt/'
if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)
# if not os.path.exists(savepath_gt):
#     os.makedirs(savepath_gt)
loop = tqdm(enumerate(UW_test_loader),total=len(UW_test_loader))

for i,(A_map,data) in loop:
    
    
    with torch.no_grad():
        
        data=data.cuda()
        A_map=A_map.cuda()


        gt = test_data[i]
        img_name = gt.split('/')[-1].split('.')[0]
        gt = Image.open(gt).convert("RGB")
        gt = transforms.functional.to_tensor(gt) 
        gt = transforms.Resize([opt.test_size,opt.test_size])(gt)
        gt = gt.unsqueeze(0)
        gt= gt.cuda()

        depth_map = test_depth[i]
        depth_map = Image.open(depth_map).convert("L")
        depth_map = transforms.functional.to_tensor(depth_map) 
        #print(depth_map)
        depth_map = transforms.Resize([opt.test_size,opt.test_size])(depth_map)
        depth_map = depth_map.unsqueeze(0)
        depth_map= depth_map.cuda()

        x = torch.cat([gt,depth_map,A_map],1)
        #start = time.time()
        g1_output = netG_1(x)

        #image_grid = torch.cat((gt,depth_map1,g1_output,data,A_map), 3)
        

        save_image(g1_output,os.path.join(opt.save_path,'%s.png'%(i+1)),normalize=False)
        #save_image(gt,'/mnt/data/csx/Documents/dataset/LNRUD/LNRUD_gt/'+ '%s.png'%(i+1),normalize=False)
    
        print('Generated images %04d of %04d' % (i+1, len(UW_test_loader)))
#calculate fid
fid = calculate_fid_given_paths([opt.save_path,opt.fid_gt_path],50,'cuda:0',2048,1)
print('fid',fid)
