import random
from models import vdsr,dncnn,unet,scUnet, wdmn13up,wdmn4,wdmn5,wdmn5l,sun_demoire,rcan_nu,rcan_nuss,DFCAN_Large
from models import wdmn6,wdmn7,wdmn8,wdmn9,wdmn10,wdmn11,wdmn12,wdmn13,wdmn133,wdmn13large2,wdmn13up,wdmn14up,wdmn15up,wdmn16up,wdmn16up2,wdmn16up2lite
from models import DFCAN,wdmn5exs,wdmn5ex2s,unet3,unet15,scUnet15,wdmn5ex,wdmn5ex2,wdmn5f,wdmn8hp
from models.modules import L1_Charbonnier_loss,L1_Sobel_Loss,L1_Wavelet_Loss,L1_ASL
import torch.nn as nn
import os

class Config_DLSIM():
    def __init__(self):
        # general
        model_name = 'WDMN8HP'  # VDSR, DNCNN, UNET, SCUNET, RCAN, WDMN5, DMCNN, DFCAN
        self.nframes = 1
        self.which_frame = 0
        self.dataset_name = 'DL-SIM'   # DL-SIM, W2S, DL-SR, STORM
        self.model = self.get_model(model_name)
        self.dataname = self.get_dataname('microtubule')  
        self.wf = False
        self.sample_large = False
        self.input_small = False
        # DL-SIM: adhesion, factin, microtubule, mitochondria
        # DL-SR: F-actin, CCPs, ER
        # W2S: W2S
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
        if self.wf:
            self.ckpt_path = 'checkpoints/%dWF%d-%s-%s-%s/'%(int(self.max_hr),self.nframes,model_name,self.datamode,self.dataname)
        else:
            self.ckpt_path = 'ablation/rev3_5block/checkpoints/%dF%d-%s-%s-%s/'%(int(self.max_hr),self.nframes,model_name,self.datamode,self.dataname)
            # self.ckpt_path = 'rebuttal/checkpoints/%dW%d-%s-%s-%s/'%(int(self.max_hr),self.which_frame,model_name,self.datamode,self.dataname)
            # self.ckpt_path = 'checkpoints/%d-%s-%s-%s/'%(int(self.max_hr),model_name,self.datamode,self.dataname)
            # self.ckpt_path = 'checkpoints/65535-WDMN4-HE-microtubule/'
        # self.ckpt_path = 'checkpoints/%d-%s-%s-%s/'%(int(self.max_hr),model_name,self.datamode,self.dataname)
        self.gpu_ids = [0,1] 
        criterion_names = ['L1']   # L1, L2, L1C, L1S, L1W, ASL
        self.criterions = self.get_criterion(criterion_names)
        
        # resume and finetuning
        resume = 0
        finetuning = True
        if resume > 0:
            self.resume = self.ckpt_path + '/model_epoch_%d.pth'%resume
        else:
            self.resume = ''
        if finetuning:
            # self.pretrained = self.ckpt_path + '/model_epoch_best.pth'
            self.pretrained = 'checkpoints/%dF%d-%s-%s-%s/'%(int(self.max_hr),self.nframes,'WDMN8',self.datamode,self.dataname) + '/model_epoch_best.pth'
            # self.pretrained = 'checkpoints/%dF%d-%s-%s-%s/'%(int(self.max_hr),15,'WDMN13',self.datamode,self.dataname) + '/model_epoch_best.pth'
            # self.pretrained = './checkpoints/65535-WDMN5-%s-%s'%(self.datamode,self.dataname)+ '/model_epoch_best.pth'
        else:
            self.pretrained = ''
        self.start_epoch = 1

        # training
        self.init_learning_rate = 0.00001
        self.lr_decay_step = 10
        self.total_epochs = 10
        self.batch_size = 12
        self.show = 50
        self.threads = 12
        self.seed = random.randint(1, 10000) #random seed for initialization

        # testing 
        self.save_image_ext = '.png'
        self.save_path = './rebuttal/ablation/%dF%d-%s-%s-%s/'%(int(self.max_hr),self.nframes,model_name,self.datamode,self.dataname)

        # A.M.P
        self.use_amp = True
    
    def get_dataname(self,name):
        if self.dataset_name == 'W2S':
            dataname = 'W2S'
        elif self.dataset_name == 'DL-SIM':
            dataname = name
        elif self.dataset_name == 'DL-SR':
            dataname = name
        return dataname
    
    def get_datamode(self,mode):
        if self.dataset_name == 'W2S' or self.dataset_name == 'DL-SR':
            datamode = 'Widefield'
        else:
            datamode = mode
        return datamode
    
    def get_path(self,mode):
        if mode == 'remote':
            if self.dataset_name == 'DL-SIM':
                datapath = '/test/sim/data/dl-sim'
            elif self.dataset_name == 'DL-SR':
                datapath = '/test/sim/data/6data/DL-SR-main/dataset/train'
            elif self.dataset_name == 'W2S':
                datapath = '/test/sim/data/w2s-raw'
        elif mode == 'local':
            datapath = '/dataset/dl-sim-clean'
        else:
            raise ValueError('use data mode must be local or remote')    
        return datapath
    
    def get_model(self,name):
        if self.dataset_name == 'DL-SIM':
            if name == 'VDSR':
                model = vdsr.Net()
            elif name == 'DNCNN':
                model = dncnn.Net()
            elif name == 'UNET' and self.nframes == 1:
                model = unet.Net()
            elif name == 'UNET' and self.nframes == 3:
                model = unet3.Net()
            elif name == 'UNET' and self.nframes == 15:
                model = unet15.Net()
            elif name == 'SCUNET' and self.nframes == 1:
                model = scUnet.Net()
            elif name == 'SCUNET' and self.nframes == 15:
                model = scUnet15.Net()
            elif name == 'DMCNN':
                model = sun_demoire.Net()
            elif name == 'RCAN':
                model = rcan_nu.RCAN(n_colors=self.nframes)
            elif name == 'WDMN4':
                model = wdmn4.Net()
            elif name == 'WDMN5' and self.nframes == 1:
                model = wdmn5.Net()     
            elif name == 'WDMN5' and self.nframes == 3:
                model = wdmn5exs.Net()     
            elif name == 'WDMN5' and self.nframes == 15:
                model = wdmn5ex2.Net()    
            elif name == 'WDMN5L': 
                model = wdmn5l.Net()   #ablation study --- no wavelet
            elif name == 'WDMN5F':
                model = wdmn5f.Net()
            elif  name == 'DFCAN':
                model = DFCAN.DFCAN(channel_in=self.nframes)
            elif  name == 'DFCANL':
                model = DFCAN_Large.DFCAN(channel_in=self.nframes)
            elif name == 'WDMN6': 
                model = wdmn6.Net(self.nframes)  
            elif name == 'WDMN7': 
                model = wdmn7.Net(self.nframes)  
            elif name == 'WDMN8': 
                model = wdmn8.Net(self.nframes)  
            elif name == 'WDMN8HP': # hyper parameter study
                model = wdmn8hp.Net(self.nframes)  
            elif name == 'WDMN9': 
                model = wdmn9.Net(self.nframes)  
            elif name == 'WDMN10': 
                model = wdmn10.Net(self.nframes)  
            elif name == 'WDMN11': 
                model = wdmn11.Net(self.nframes) 
            elif name == 'WDMN12': 
                model = wdmn12.Net(self.nframes) 
            elif name == 'WDMN13': 
                model = wdmn13.Net(self.nframes) 
            elif name == 'WDMN133': 
                model = wdmn133.Net(self.nframes)  
            elif name == 'WDMN13L': 
                model = wdmn13large2.Net(self.nframes)  
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







