import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
import torch,warnings
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',type=bool,default=True)
parser.add_argument('--shutil',type=bool,default=False)
parser.add_argument('--eval_epoch',type=int,default=1)
parser.add_argument('--test_epoch',type=int,default=10)

parser.add_argument('--cuda', action='store_true',default=True, help='use GPU computation')
parser.add_argument('--crop',action='store_true',default=True)
parser.add_argument('--crop_size',type=int,default=256,help='Takes effect when using --crop ')
parser.add_argument('--crop_size_test',type=str,default=256,help='Takes effect when using --crop ')

parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--bs', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
					help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')

parser.add_argument('--model_store',type=str,default='./train_models/',help='the path to save model') 
parser.add_argument('--model_name',type=str,default='UWNR')

parser.add_argument('--tensorboardX_path',type=str,default='./runs/',help='the path to save tensorboardX') 

parser.add_argument('--train_save_path',type=str,default='./train_out',help='the path to save generation underwater image')
parser.add_argument('--clean_img_path',type=str,default='/mnt/data/csx/Documents/underwater_generation2022/NYU_test/NYU_GT/',help='the path of saving clean image')
parser.add_argument('--depth_img_path',type=str,default='/mnt/data/csx/Documents/underwater_generation2022/NYU_test/NYU_depth/',help='the path of saving depth image')
parser.add_argument('--underwater_path',type=str,default='/mnt/data/csx/Documents/dataset/UIEB/',help='the path of saving real underwater image')
parser.add_argument('--fid_gt_path',type=str,default='/mnt/data/csx/Documents/underwater_generation2022/train_models/UIEB_256',help='the path of real underwater image as fid ground truth')
#ddp
parser.add_argument("--local_rank", default=-1, type=int)

opt = parser.parse_args()

if not os.path.exists(opt.model_store):
 	os.makedirs(opt.model_store,exist_ok=True)
if not os.path.exists(opt.model_store+opt.model_name):
 	os.makedirs(opt.model_store+opt.model_name,exist_ok=True)
opt.model_dir = opt.model_store + opt.model_name+ '/'
print(opt.model_dir)
