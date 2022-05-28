import torch.utils.data as data
import torch
import numpy as np
from skimage import transform,measure,io
import os
from PIL import Image
from torchvision import transforms
import random
import config
cfg = config.Config()

def data_augment(im,num):
    # org_image = im.transpose(1,2,0)
    org_image = im
    if num ==0:    
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num ==1:      
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num ==2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)        
        tranform = lrud_image
    elif num ==3:
        rotated_image1 = np.rot90(org_image)        
        tranform = rotated_image1
    elif num ==4: 
        rotated_image2 = np.rot90(org_image, -1)
        tranform = rotated_image2
    elif num ==5: 
        rotated_image1 = np.rot90(org_image) 
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num ==6:        
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    # tranform = tranform.transpose(2,0,1)
    return tranform

def random_crop(moire,clean,crop_size=256,fixed_pos = None):
    #暂时不用
    moire = moire/cfg.max_lr
    clean = clean/cfg.max_hr
    im_size = clean.shape[0]   #h,w must be same, moire->512x512, clean 1024x1024
    # scale = clean.shape[0] // moire.shape[0]
    moire = transform.resize(moire,[clean.shape[0],clean.shape[1]],order=3) # BICUBIC
    if crop_size==im_size:
        # moire = moire.resize((256, 256),Image.BICUBIC)
        nm = moire
        nc = clean
    else:
        if fixed_pos == None:
            rand_num_x = np.random.randint(im_size-crop_size-1)
            rand_num_y = np.random.randint(im_size-crop_size-1)
        else:
            rand_num_x = fixed_pos[0]
            rand_num_y = fixed_pos[1]
        
        moire = np.array(moire)
        clean = np.array(clean)
        nm = moire[rand_num_x:rand_num_x+crop_size,rand_num_y:rand_num_y+crop_size]
        nc = clean[rand_num_x:rand_num_x+crop_size,rand_num_y:rand_num_y+crop_size]
    nm = torch.from_numpy(nm.astype(np.float32)).view(1,crop_size,crop_size)
    nc = torch.from_numpy(nc.astype(np.float32)).view(1,crop_size,crop_size)
    return nm,nc

class DatasetFromImage(data.Dataset):
    def __init__(self, dataname):
        super(DatasetFromImage, self).__init__()
        # root = '../../data/dl-sim/%s/Training_Testing_%s/'%(dataname,dataname)
        if cfg.dataset_name == 'DL-SIM':
            root = '%s/%s/Training_Testing_%s/'%(cfg.data_path,dataname,dataname)
            if cfg.input_small == False:
                if cfg.datamode == 'HE':
                    MHR,MLR = ['%s/HER'%root,'%s/HE_X2'%root]
                elif cfg.datamode == 'LE':
                    MHR,MLR = ['%s/HER'%root,'%s/LE_X2'%root]
            if cfg.input_small == True:
                if cfg.datamode == 'HE':
                    MHR,MLR = ['%s/HER'%root,'%s/HE'%root]
                elif cfg.datamode == 'LE':
                    MHR,MLR = ['%s/HER'%root,'%s/LE'%root]                
            self.HR = os.path.join(MHR)
            self.LR = os.path.join(MLR)
            self.HR_list = os.listdir(self.HR) # sample1.tif
            self.LR_list = os.listdir(self.LR) # sample1/HE_00.tif
            # self.crop_size = 256
            if cfg.input_small:
                if cfg.datamode == 'LE':
                    self.title_str = 'LE'
                else:
                    self.title_str = 'HE'   
            else:  
                if dataname == 'microtubule' and cfg.datamode == 'LE':
                    self.title_str = 'LE'
                else:
                    self.title_str = 'HE'   
            self.data_augment = cfg.data_augment   
        elif cfg.dataset_name == 'W2S':
            self.base_gt = os.path.join(cfg.data_path,'train_gt_level1')
            self.base_wf = os.path.join(cfg.data_path,'train_wf_level1')
            self.HRLR_dirs =  os.listdir(self.base_gt)
            self.HR_list = []
            for idir in self.HRLR_dirs:
                images = os.listdir(os.path.join(self.base_gt, idir))
                for image in images:
                    self.HR_list.append(os.path.join(idir, image))
        elif cfg.dataset_name == 'DL-SR':
            self.base = os.path.join(cfg.data_path,cfg.dataname)
            self.HR = os.path.join(self.base,'training_gt')
            self.LR = os.path.join(self.base,'training_wf')
            self.HR_list = os.listdir(os.path.join(self.base,'training_gt'))
            self.LR_list = os.listdir(os.path.join(self.base,'training_wf'))
        elif cfg.dataset_name == 'DL-SR2':
            self.data_augment = cfg.data_augment   
            self.base = os.path.join(cfg.data_path,cfg.dataname)
            self.HR = os.path.join(self.base,'training_gt')
            self.LR = os.path.join(self.base,'training')
            self.HR_list = os.listdir(os.path.join(self.base,'training_gt'))
            self.LR_list = os.listdir(os.path.join(self.base,'training'))

    def dlsr_item(self, index):
        HR = io.imread(os.path.join(os.path.join(self.base,'training_gt'),self.HR_list[index]))
        THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
        LR = io.imread(os.path.join(os.path.join(self.base,'training_wf'),self.HR_list[index]))
        if cfg.input_small:
            TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,128,128)
        else:
            LR = transform.resize(LR/65535.0,[256,256],order=3)*65535.0
            TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,256,256)
        return TLR, THR 

    def w2s_item(self, index):
        sample = self.HR_list[index]
        wf = io.imread(os.path.join(self.base_wf,sample))
        gt = io.imread(os.path.join(self.base_gt,sample))
        # WF = np.load(os.path.join(self.base_wf, sample_folder,'wf_channel0.npy'))   # 400x512x512
        # nWF = np.zeros([512,512])
        # for i in range(cfg.nframes):
        #     nWF += WF[i,:,:]
        # nWF = nWF/cfg.nframes
        # SIM_gt = np.load(os.path.join(self.base_gt, sample_folder,'sim_channel0.npy'))   #1024x1024
        # TLR, THR = random_crop(nWF,SIM_gt)
        TLR = torch.from_numpy(wf.astype(np.float32)/cfg.max_lr).view(1,256,256)
        THR = torch.from_numpy(gt.astype(np.float32)/cfg.max_hr).view(1,256,256)
        return TLR, THR 

    def dlsim_item_saparate(self, index):
        HR = io.imread(os.path.join(self.HR,self.HR_list[index]))
        no_frame =cfg.which_frame
        if cfg.nframes == 1:
            if cfg.input_small == False:
                rand_index = no_frame
                if rand_index < 10:
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_0%d.tif'%(self.title_str,rand_index))
                else:
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,rand_index))
            else:
                rand_index = np.random.randint(1,16)
                local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,rand_index))                  
            LR = io.imread(os.path.join(self.LR,local_LR_path))

            if self.data_augment:
                aug_num = np.random.randint(0, 8)
                HR = data_augment(HR,aug_num) 
                LR = data_augment(LR,aug_num)   
            if cfg.input_small == False:
                THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
                TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,256,256)
            else:
                THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
                TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,128,128)                    
        return TLR, THR

    def dlsim_item(self, index):
        HR = io.imread(os.path.join(self.HR,self.HR_list[index]))
        if cfg.nframes == 1:
            if cfg.wf == False:
                # random choose a LR
                if cfg.input_small == False:
                    rand_index = np.random.randint(0,15)
                    if rand_index < 10:
                        local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_0%d.tif'%(self.title_str,rand_index))
                    else:
                        local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,rand_index))
                else:
                    rand_index = np.random.randint(1,16)
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,rand_index))                  
                LR = io.imread(os.path.join(self.LR,local_LR_path))

                if self.data_augment:
                    aug_num = np.random.randint(0, 8)
                    HR = data_augment(HR,aug_num) 
                    LR = data_augment(LR,aug_num)   
                if cfg.input_small == False:
                    THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
                    TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,256,256)
                else:
                    THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
                    TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,128,128)                    
            else:
                LR = np.zeros([1,256,256])
                for nf in range(15):
                    if nf < 10:
                        local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_0%d.tif'%(self.title_str,nf))
                    else:
                        local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,nf))
                    LR_frame = io.imread(os.path.join(self.LR,local_LR_path))
                    LR[0,:,:] += LR_frame
                LR = LR/15
                THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
                TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,256,256)
        elif cfg.nframes == 3:
            LR = np.zeros([3,256,256])
            read_type = 1 #  sequence or phase
            if  read_type == 1:
                for i in range(3):
                    ii=i*5
                    if ii <= 9:
                        local_LR_path = os.path.join(self.HR_list[index][:-4], "%s_0"%self.title_str+str(ii)+".tif")
                    else:
                        local_LR_path = os.path.join(self.HR_list[index][:-4], "%s_"%self.title_str+str(ii)+".tif")
                    LR_frame = io.imread(os.path.join(self.LR,local_LR_path))
                    LR[i,:,:] = LR_frame
            else:
                for nf in range(cfg.nframes):
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_0%d.tif'%(self.title_str,nf))
                    LR_frame = io.imread(os.path.join(self.LR,local_LR_path))
                    LR[nf,:,:] = LR_frame
            THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
            TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(3,256,256)
        elif cfg.nframes == 15:
            LR = np.zeros([15,256,256])
            for nf in range(cfg.nframes):
                if nf < 10:
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_0%d.tif'%(self.title_str,nf))
                else:
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,nf))
                LR_frame = io.imread(os.path.join(self.LR,local_LR_path))
                LR[nf,:,:] = LR_frame
            THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
            TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(15,256,256)
        return TLR, THR




    def dlsr2_item(self, index):
        HR = io.imread(os.path.join(self.HR,self.HR_list[index]))
        if cfg.nframes == 1:
            # random choose a LR
            rand_index = np.random.randint(1,9)
            local_LR_path = os.path.join(self.HR_list[index][:-4],'%d.tif'%rand_index)
            LR = io.imread(os.path.join(self.LR,local_LR_path))
            LR = transform.resize(LR,(256,256),order=3)

            if self.data_augment:
                aug_num = np.random.randint(0, 8)
                HR = data_augment(HR,aug_num) 
                LR = data_augment(LR,aug_num)   
            THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
            TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,256,256)
        return TLR, THR


    def __getitem__(self, index):
        if cfg.dataset_name == 'DL-SIM' and cfg.which_frame == None:
            TLR, THR = self.dlsim_item(index)
        elif cfg.dataset_name == 'DL-SIM' and cfg.which_frame != None:
            TLR, THR = self.dlsim_item_saparate(index)
        elif cfg.dataset_name == 'W2S':
            TLR, THR = self.w2s_item(index)
        elif cfg.dataset_name == 'DL-SR':
            TLR, THR = self.dlsr_item(index)
        elif cfg.dataset_name == 'DL-SR2':
            TLR, THR = self.dlsr2_item(index)
        return TLR, THR

        
    def __len__(self):
        return len(self.HR_list)

class TestLoader(data.Dataset):
    def __init__(self, dataname):
        super(TestLoader, self).__init__()
        if cfg.dataset_name == 'DL-SIM':
            root = '%s/%s/Training_Testing_%s/'%(cfg.data_path,dataname,dataname)
            if cfg.input_small == False:
                if cfg.datamode == 'HE':
                    MHR,MLR = ['%s/testing_HER'%root,'%s/testing_HE_X2'%root]
                elif cfg.datamode == 'LE':
                    MHR,MLR = ['%s/testing_HER'%root,'%s/testing_LE_X2'%root]
            if cfg.input_small == True:
                if cfg.datamode == 'HE':
                    MHR,MLR = ['%s/testing_HER'%root,'%s/testing_HE'%root]
                elif cfg.datamode == 'LE':
                    MHR,MLR = ['%s/testing_HER'%root,'%s/testing_LE'%root]                
            self.HR = os.path.join(MHR)
            self.LR = os.path.join(MLR)
            self.HR_list = os.listdir(self.HR) # sample1.tif
            self.LR_list = os.listdir(self.LR) # sample1/HE_00.tif
            if cfg.input_small:
                if cfg.datamode == 'LE':
                    self.title_str = 'LE'
                else:
                    self.title_str = 'HE'   
            else:  
                if dataname == 'microtubule' and cfg.datamode == 'LE':
                    self.title_str = 'LE'
                else:
                    self.title_str = 'HE'   
            # self.data_augment = cfg.data_augment  # Test暂不使用~self-ensemble后期再说 
        elif cfg.dataset_name == 'DL-SR':
            self.base = os.path.join(cfg.data_path,cfg.dataname)
            self.HR = os.path.join(self.base,'validate_gt')
            self.LR = os.path.join(self.base,'validate_wf')
            self.HR_list = os.listdir(os.path.join(self.base,'validate_gt'))
            self.LR_list = os.listdir(os.path.join(self.base,'validate_wf'))

    def dlsr_item(self, index):
        HR = io.imread(os.path.join(os.path.join(self.base,'validate_gt'),self.HR_list[index]))
        THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
        LR = io.imread(os.path.join(os.path.join(self.base,'validate_wf'),self.HR_list[index]))
        if cfg.input_small:
            TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,128,128)
        else:
            LR = transform.resize(LR/65535.0,[256,256],order=3)*65535.0
            TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,256,256)
        return TLR, THR 

    def dlsim_item(self, index):
        HR = io.imread(os.path.join(self.HR,self.HR_list[index]))
        if cfg.nframes == 1:
            if cfg.wf == False:
                if cfg.input_small == False:
                    rand_index = np.random.randint(0,15)
                    if rand_index < 10:
                        local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_0%d.tif'%(self.title_str,rand_index))
                    else:
                        local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,rand_index))
                else:
                    rand_index = np.random.randint(1,16)
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,rand_index))                  
                LR = io.imread(os.path.join(self.LR,local_LR_path))

                if cfg.input_small == False:
                    THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
                    TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,256,256)
                else:
                    THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
                    TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,128,128)                    
            else:
                LR = np.zeros([1,256,256])
                for nf in range(15):
                    if nf < 10:
                        local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_0%d.tif'%(self.title_str,nf))
                    else:
                        local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,nf))
                    LR_frame = io.imread(os.path.join(self.LR,local_LR_path))
                    LR[0,:,:] += LR_frame
                LR = LR/15
                THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
                TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,256,256)
        elif cfg.nframes == 3:
            LR = np.zeros([3,256,256])
            read_type = 1 #  sequence or phase
            if  read_type == 1:
                for i in range(3):
                    ii=i*5
                    if ii <= 9:
                        local_LR_path = os.path.join(self.HR_list[index][:-4], "%s_0"%self.title_str+str(ii)+".tif")
                    else:
                        local_LR_path = os.path.join(self.HR_list[index][:-4], "%s_"%self.title_str+str(ii)+".tif")
                    LR_frame = io.imread(os.path.join(self.LR,local_LR_path))
                    LR[i,:,:] = LR_frame
            else:
                for nf in range(cfg.nframes):
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_0%d.tif'%(self.title_str,nf))
                    LR_frame = io.imread(os.path.join(self.LR,local_LR_path))
                    LR[nf,:,:] = LR_frame
            THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
            TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(3,256,256)
        elif cfg.nframes == 15:
            LR = np.zeros([15,256,256])
            for nf in range(cfg.nframes):
                if nf < 10:
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_0%d.tif'%(self.title_str,nf))
                else:
                    local_LR_path = os.path.join(self.HR_list[index][:-4],'%s_%d.tif'%(self.title_str,nf))
                LR_frame = io.imread(os.path.join(self.LR,local_LR_path))
                LR[nf,:,:] = LR_frame
            THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
            TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(15,256,256)
        return TLR, THR


    def __getitem__(self, index):
        if cfg.dataset_name == 'DL-SIM':
            TLR, THR = self.dlsim_item(index)
        elif cfg.dataset_name == 'W2S':
            TLR, THR = self.w2s_item(index)
        elif cfg.dataset_name == 'DL-SR':
            TLR, THR = self.dlsr_item(index)
        elif cfg.dataset_name == 'DL-SR2':
            TLR, THR = self.dlsr2_item(index)
        return TLR, THR

        
    def __len__(self):
        return len(self.HR_list)


def test():
    file_path = "./"
    dfi = DatasetFromImage(file_path)
