import random
from models import vdsr,dncnn,unet,scUnet,wdmn4,wdmn5,wdmn5l,sun_demoire,rcan_nu,rcan_nuss,wdmn6,wdmn7,wdmn8,wdmn9,wdmn10,wdmn13
from models import DFCAN,wdmn5exs,wdmn5ex2s,unet3,unet15,scUnet15,wdmn5ex,wdmn5ex2,wdmn5f,DFCAN_Large
from models import wdmn6,wdmn7,wdmn8,wdmn9,wdmn10,wdmn11,wdmn12,wdmn13,wdmn133,wdmn13large2,wdmn13up,wdmn14up,wdmn15up,wdmn16up,wdmn16up2,wdmn16up2lite
from models.modules import L1_Charbonnier_loss,L1_Sobel_Loss,L1_Wavelet_Loss,L1_ASL
import torch.nn as nn
import os

class Config_DLSR():
    def __init__(self):
        # general
        model_name = 'UNET'  # VDSR, DNCNN, UNET, SCUNET, RCAN, WDMN5, DMCNN, DFCAN
        self.nframes = 1
        self.dataset_name = 'DL-SR'   # DL-SIM, W2S, DL-SR, STORM
        self.model = self.get_model(model_name)
        self.dataname = self.get_dataname('F-actin')  
        # DL-SIM: adhesion, factin, microtubule, mitochondria
        # DL-SR: F-actin, CCPs, ER
        # W2S: W2S
        self.sample_large = True
        self.input_small = False
        self.datamode = self.get_datamode('HE')
        use_data = 'remote' #local
        self.data_path = self.get_path(use_data)
        # self.max_hr = 15383.0
        # self.max_lr = 5315.0
        self.max_hr = 65535.0     # W2S: 39972
        self.max_lr = 65535.0       
        self.max_range = 65535.0
        self.max_range_rgb = 255.0
        self.data_augment = False
        self.ckpt_path = 'checkpoints/%dF%d-%s-%s-%s/'%(int(self.max_hr),self.nframes,model_name,self.datamode,self.dataname)
        self.gpu_ids = [0,1,2] 
        criterion_names = ['L1']   # L1, L2, L1C, L1S, L1W, ASL
        self.criterions = self.get_criterion(criterion_names)
        
        # resume and finetuning
        resume = 0
        finetuning = False
        if resume > 0:
            self.resume = self.ckpt_path + '/model_epoch_%d.pth'%resume
        else:
            self.resume = ''
        if finetuning:
            # self.pretrained = self.ckpt_path + '/model_epoch_best.pth'
            self.pretrained = 'checkpoints/%dF%d-%s-%s-%s/'%(int(self.max_hr),self.nframes,'WDMN16UP2LT','HE','factin') + '/model_epoch_best.pth'
            # self.pretrained = './checkpoints/65535-WDMN5-%s-%s'%(self.datamode,self.dataname)+ '/model_epoch_best.pth'
        else:
            self.pretrained = ''
        self.start_epoch = 1

        # training
        self.init_learning_rate = 0.0001
        self.lr_decay_step = 40
        self.total_epochs = 150
        self.batch_size = 20
        self.show = 800
        self.threads = 10
        self.seed = random.randint(1, 10000) #random seed for initialization

        # testing 
        self.save_image_ext = '.png'
        self.save_path = './bw/DLSR/DLSR-max-%s-%s-%s/'%(model_name,self.datamode,self.dataname)

        # A.M.P
        self.use_amp = False
    
    def get_dataname(self,name):
        if self.dataset_name == 'DL-SR':
            dataname = name
        return dataname
    
    def get_datamode(self,mode):
        if self.dataset_name == 'W2S' or self.dataset_name == 'DL-SR':
            datamode = 'Widefield'
        else:
            datamode = mode
        return datamode
    
    def get_path(self,mode):
        if self.dataset_name == 'DL-SR':
            datapath = '/test/sim/data/6data/DL-SR-main/dataset/train'
        return datapath
    
    def get_model(self,name):
        if name == 'VDSR':
            model = vdsr.Net()
        elif name == 'DNCNN':
            model = dncnn.Net() 
        elif name == 'DMCNN':
            model = sun_demoire.Net()           
        elif name == 'UNET':
            model = unet.Net()
        elif name == 'SCUNET':
            model = scUnet.Net()
        elif name == 'RCAN':
            model = rcan_nu.RCAN(n_colors=1)
        elif name == 'RCANS':
            model = rcan_nuss.RCAN(n_colors=1)
        elif  name == 'DFCAN':
            model = DFCAN.DFCAN(channel_in=1)
        elif  name == 'DFCANL':
            model = DFCAN_Large.DFCAN(channel_in=1)
        elif name == 'WDMN8': 
            model = wdmn8.Net(self.nframes)  
        elif name == 'WDMN9': 
            model = wdmn9.Net(self.nframes)  
        elif name == 'WDMN10': 
            model = wdmn10.Net(self.nframes)  
        elif name == 'WDMN13': 
            model = wdmn13.Net(self.nframes)  
        elif name == 'WDMN13UP': 
            model = wdmn13up.Net(self.nframes)  
        elif name == 'WDMN14UP': 
            model = wdmn14up.Net(self.nframes)  
        elif name == 'WDMN15UP': 
            model = wdmn15up.Net(self.nframes) 
        elif name == 'WDMN16UP': 
            model = wdmn16up.Net(self.nframes) 
        elif name == 'WDMN16UP2': 
            model = wdmn16up2.Net(self.nframes) 
        elif name == 'WDMN16UP2LT': 
            model = wdmn16up2lite.Net(self.nframes) 
        return model
    
    def get_criterion(self,list_names):
        list_criterions = []
        for name in list_names:
            if name == 'L1':
                list_criterions.append(nn.L1Loss())
            elif name == 'L2':
                list_criterions.append(nn.MSELoss())
            elif name == 'L1C':
                list_criterions.append(L1_Charbonnier_loss())
            elif name == 'L1S':
                list_criterions.append(L1_Sobel_Loss(in_channels=1))
            elif name == 'L1W':
                list_criterions.append(L1_Wavelet_Loss())
            elif name == 'ASL':
                list_criterions.append(L1_ASL())
        return list_criterions







