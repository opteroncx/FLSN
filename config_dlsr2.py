import random
from models import vdsr,dncnn,unet,scUnet,wdmn4,wdmn5,wdmn5l,sun_demoire,rcan_nu,rcan_nuss,wdmn6,wdmn7,wdmn8,wdmn9,wdmn10,wdmn13
from models import DFCAN,wdmn5exs,wdmn5ex2s,unet3,unet15,scUnet15,wdmn5ex,wdmn5ex2,wdmn5f
from models.modules import L1_Charbonnier_loss,L1_Sobel_Loss,L1_Wavelet_Loss,L1_ASL
import torch.nn as nn
import os

class Config_DLSR2():
    def __init__(self):
        # general
        model_name = 'UNET'  # VDSR, DNCNN, UNET, SCUNET, RCAN, WDMN5, DMCNN, DFCAN
        self.nframes = 1
        self.dataset_name = 'DL-SR2'   # DL-SIM, W2S, DL-SR, STORM
        self.model = self.get_model(model_name)
        self.dataname = self.get_dataname('F-actin')  
        # DL-SIM: adhesion, factin, microtubule, mitochondria
        # DL-SR: F-actin, CCPs, ER
        # W2S: W2S
        self.datamode = self.get_datamode('HE')
        use_data = 'remote' #local
        self.data_path = self.get_path(use_data)
        self.max_hr = 65535.0     # W2S: 39972
        self.max_lr = 65535.0       
        self.max_range = 65535.0
        self.max_range_rgb = 255.0
        self.data_augment = False
        self.ckpt_path = 'checkpoints/%dF%d-%s-%s-%s/'%(int(self.max_hr),self.nframes,model_name,self.datamode,self.dataname)
        self.gpu_ids = [0,1,2,3] 
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
            self.pretrained = 'checkpoints/0dlsr/%dF%d-%s-%s-%s/'%(int(self.max_hr),self.nframes,'WDMN10',self.datamode,self.dataname) + '/model_epoch_best.pth'
            # self.pretrained = './checkpoints/65535-WDMN5-%s-%s'%(self.datamode,self.dataname)+ '/model_epoch_best.pth'
        else:
            self.pretrained = ''
        self.start_epoch = 1

        # training
        self.init_learning_rate = 0.0001
        self.lr_decay_step = 40
        self.total_epochs = 150
        self.batch_size = 64
        self.show = 800
        self.threads = 8
        self.seed = random.randint(1, 10000) #random seed for initialization

        # testing 
        self.save_image_ext = '.png'
        self.save_path = './result/65535-max-%s-%s-%s/'%(model_name,self.datamode,self.dataname)

        # A.M.P
        self.use_amp = False
    
    def get_dataname(self,name):
        if self.dataset_name == 'DL-SR2':
            dataname = name
        return dataname
    
    def get_datamode(self,mode):
        datamode = mode
        return datamode
    
    def get_path(self,mode):
        if mode == 'remote':
            if self.dataset_name == 'DL-SR2':
                datapath = '/test/sim/data/6data/DL-SR-main/dataset/train'
        else:
            raise ValueError('use data mode must be local or remote')    
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
        elif name == 'WDMN5':
            model = wdmn5.Net()   
        elif name == 'WDMN5L': 
            model = wdmn5l.Net()  
        elif name == 'WDMN6': 
            model = wdmn6.Net(self.nframes) 
        elif name == 'WDMN7': 
            model = wdmn7.Net(self.nframes)   
        elif  name == 'DFCAN':
            model = DFCAN.DFCAN(channel_in=1)
        elif name == 'WDMN8': 
            model = wdmn8.Net(self.nframes)  
        elif name == 'WDMN9': 
            model = wdmn9.Net(self.nframes)  
        elif name == 'WDMN10': 
            model = wdmn10.Net(self.nframes)  
        elif name == 'WDMN13': 
            model = wdmn13.Net(self.nframes)  
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







