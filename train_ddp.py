import os
from myutils.dataloader import UWS_Dataset_Retinex, UWS_Dataset_Retinex_test
from torchvision.utils import save_image,make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from pytorch_fid_test.src.pytorch_fid.fid_score import *
from time import time
from tqdm import *
from option import opt
import time
from model.FSU2 import *
from loss.perceptual import PerceptualLoss2,PerceptualLoss
from loss.DCLoss import DCLoss_Two
from tensorboardX import SummaryWriter
from option import *
from metrics import *
from myutils.utils import *
import shutil
import glob
from ptflops import get_model_complexity_info
#DDP
from apex import amp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from loss.retinex_loss import *
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# python3  -m torch.distributed.launch --master_port 42563 --nproc_per_node 2 train_ddp.py --resume=True
#ddp
torch.cuda.set_device(opt.local_rank)
dist.init_process_group(backend='nccl')

opt.shutil = True
if opt.local_rank == 0:
    if opt.shutil == True:
        #cover the previous tensorboardX file
        file_path_runs = glob.glob(os.path.join(opt.tensorboardX_path,f'*{opt.model_name}'))
        for i in file_path_runs:
            shutil.rmtree(i)
    
    writer = SummaryWriter(os.path.join(opt.tensorboardX_path,opt.model_name))

def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

rank = torch.distributed.get_rank()

init_seeds(41 + rank)


def train(netG_A2B,loader_train,loader_test,test_data,test_depth):

    start_epoch=0
    min_fid = 1000

    vgg_loss =PerceptualLoss().cuda()
    dc_loss = DCLoss_Two(15).cuda()
    lrc = torch.nn.L1Loss()
    retinex_loss = Retinex_loss1().cuda()
    


    flops_t, params_t = get_model_complexity_info(netG_A2B, (7, opt.crop_size, opt.crop_size), as_strings=True, print_per_layer_stat=True)
    
    print(f"net flops:{flops_t} parameters:{params_t}")
    


    optimizer_G1 = torch.optim.AdamW(netG_A2B.parameters(), lr=opt.lr,betas=[0.9,0.999],weight_decay=0.000001)

    netG_A2B,optimizer_G1= amp.initialize(netG_A2B,optimizer_G1,opt_level='O1')

    netG_A2B = DDP(netG_A2B,delay_allreduce=True)


    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G1,
    												lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    
    if opt.resume and os.path.exists(opt.model_dir+'utils_'+opt.model_name+'.pk'):
        ckp=torch.load(opt.model_dir+'utils_'+opt.model_name+'.pk')
        start_epoch=ckp['epoch']
        min_fid = ckp['min_fid']
        step = ckp['step']
        if opt.local_rank == 0:
            print(f'resume from {opt.model_dir}')
            print(f'*******min FID{min_fid:.4f}*******')
            G1 = torch.load(opt.model_dir+f'net_render-epoch{start_epoch:02d}-fid{min_fid:.2f}.pk')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in G1['G1'].items():
                name = k 
                new_state_dict[name] = v
            netG_A2B.load_state_dict(new_state_dict)
            new_state_dict = OrderedDict()

            optimizer_G1.load_state_dict(G1['optimizer_weightG1'])
    else :
        step =0
        if opt.local_rank == 0:

            netG_A2B.apply(weights_init_normal)
            
            start_epoch = 0
            print('train from scratch *** ')

    #**************train****************#
    step = step
    for epoch in range(start_epoch+1, opt.n_epochs+1):
        netG_A2B.train()
        loop = tqdm(enumerate(loader_train),total=len(loader_train))
        loader_train.sampler.set_epoch(epoch)
        for i, batch in loop:
            step += 1

            (data,target,depth_map,A_map) = batch
            data=data.cuda();target=target.cuda();depth_map=depth_map.cuda();A_map=A_map.cuda()

            x = torch.cat([target,depth_map,A_map],1) 
            I_0 = netG_A2B(x)
   
            reti_loss_l1 = retinex_loss(data,I_0)
            reti_loss_l1 = 0.5*reti_loss_l1
            vg_loss = 0.5*(vgg_loss(data,I_0))
            t_loss =  0.5*(dc_loss(data[:,1:],I_0[:,1:]))
            lrcloss = 0.6*(lrc(data,I_0))
            loss =t_loss+vg_loss+lrcloss+reti_loss_l1

            optimizer_G1.zero_grad()
            with amp.scale_loss(loss,optimizer_G1)as scale_losss:
               scale_losss.backward()
            optimizer_G1.step()

            if opt.local_rank == 0:
                writer.add_scalar('epoch',epoch,step)
                writer.add_scalar('loss',loss.item(),step)
                writer.add_scalar('dc_loss',t_loss.item(),step)
                writer.add_scalar('lrcloss',lrcloss.item(),step)
                writer.add_scalar('reti_loss_pe',reti_loss_l1.item(),step) 
                writer.add_scalar('vgloss',vg_loss.item(),step)
                if i % 2 == 0 :
                    print(f'\n|loss : {loss.item():.5f} |epoch :{epoch}/{opt.n_epochs} |step :{i*2}/{len(loader_train)*2}')
            
        torch.distributed.barrier()
        # test
        if opt.local_rank == 0:
            if (epoch) % opt.test_epoch ==0 : 
                netG_A2B.eval()
                with torch.no_grad():
                    len1,fid=test(netG_A2B,loader_test,test_data,test_depth,epoch)

                    print(f'\nepoch :{epoch} |fid:{fid:.4f} |len:{len1:.4f}')
                    writer.add_scalar('fid',fid,step)
                # save every epoch model
                file_path = glob.glob(opt.model_dir+r'net*')
                if file_path :
                        os.remove(file_path[0])
                torch.save({
                            'G1':netG_A2B.state_dict(),
                            'optimizer_weightG1': optimizer_G1.state_dict(),
                            
                    },opt.model_dir+f'net_render-epoch{epoch:02d}-fid{fid:.2f}.pk')
                # save the min_fid model
                if (fid < min_fid): 
                        min_fid = fid
                        torch.save({
                                'min_fid':min_fid,
                                'epoch':epoch,
                                'step':step
                        },opt.model_dir+'utils_'+opt.model_name+'.pk')
                        file_path = glob.glob(opt.model_dir+r'sota*')
                        if file_path :
                                os.remove(file_path[0])
                        torch.save({
                                'G1':netG_A2B.state_dict(),
                                'optimizer_weightG1': optimizer_G1.state_dict(),
                                
                        },opt.model_dir+f'sota_render-epoch{epoch:02d}-fid{min_fid:.2f}.pk')

                        print(f'\n model saved at step :{epoch}| min_fid:{min_fid:.4f}')
                # save fixed epoch model
                if (epoch % 40)== 0 and (epoch>=80):
                    torch.save({
                                'G1':netG_A2B.state_dict(),
                                'optimizer_weightG1': optimizer_G1.state_dict(),
                                
                        },opt.model_dir+f'epoch_render-epoch{epoch:02d}-fid{fid:.2f}.pk')
        torch.distributed.barrier()
        # Update learning rates
        lr_scheduler_G.step()
    writer.close()
def test(G1,loader_test,test_data,test_depth,epoch):
    G1.eval()
    torch.cuda.empty_cache()

    loop = tqdm(enumerate(loader_test),total=len(loader_test))
    for i ,(A_map,data) in loop:
        A_map=A_map.cuda()
        #print(A_map.shape)

        gt = test_data[i]
        img_name = gt.split('/')[-1].split('.')[0]
        gt = Image.open(gt).convert("RGB")
        gt = transforms.functional.to_tensor(gt) 
        gt = transforms.Resize([256,256])(gt)
        gt = gt.unsqueeze(0)
        gt= gt.cuda()
        #print('gt',gt)
        depth_map = test_depth[i]
        depth_map = Image.open(depth_map).convert("L")
        depth_map = transforms.functional.to_tensor(depth_map) 
        depth_map = transforms.Resize([256,256])(depth_map)
        #print(depth_map)
        depth_map = depth_map.unsqueeze(0)
        depth_map= depth_map.cuda()
        
        x = torch.cat([gt,depth_map,A_map],1)
        g1_output = G1(x)

        save_image(g1_output,os.path.join(opt.train_save_path,'%s.png'%(img_name)),normalize=False)

    #calculate fid
    fid = calculate_fid_given_paths([opt.train_save_path,opt.fid_gt_path],50,'cuda:0',2048,1)
    #print(fid)

    return len(loader_test),fid
if __name__ == "__main__":
    os.makedirs(opt.train_save_path,exist_ok=True)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=UWS_Dataset_Retinex(opt.underwater_path,train=True,size=opt.crop_size,dcp=True),shuffle=True,num_replicas=2,rank=opt.local_rank)
    
    UW_train_loader = DataLoader(dataset=UWS_Dataset_Retinex(opt.underwater_path,train=True,size=opt.crop_size,dcp=False),batch_size=opt.bs,num_workers=0,sampler =train_sampler)
    UW_test_loader = DataLoader(dataset=UWS_Dataset_Retinex_test(opt.underwater_path,train=False,size=opt.crop_size_test,dcp=False),batch_size=1,shuffle=False,num_workers=0)
    #clean image
    test_lir=os.listdir(opt.clean_img_path)
    test_lir.sort()
    test_data =[os.path.join(opt.clean_img_path,test_img)  for test_img in test_lir]
    #clean image depthmap
    test_lir_depth=os.listdir(opt.depth_img_path)
    test_lir_depth.sort()
    test_depth =[os.path.join(opt.depth_img_path,test_img)  for test_img in test_lir_depth]
    #MHBnet
    model = Generator().cuda()

    train(model,UW_train_loader,UW_test_loader,test_data,test_depth)