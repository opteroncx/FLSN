import torch.utils.data as data
import torch
import numpy as np
import h5py
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

def random_crop(moire,clean,crop_size,im_size=1024):
    #暂时不用
    if crop_size==im_size:
        moire = moire.resize((256, 256),Image.BICUBIC)
        # print(moire)
        # moire = transform.resize(moire,[256,256],order=3)
        # moire = Image.fromarray(moire)
        return moire,clean
    else:
        rand_num_x = np.random.randint(im_size-crop_size-1)
        rand_num_y = np.random.randint(im_size-crop_size-1)
        moire = np.array(moire)
        clean = np.array(clean)
        nm = moire[rand_num_x:rand_num_x+crop_size,rand_num_y:rand_num_y+crop_size,:]
        nc = clean[rand_num_x:rand_num_x+crop_size,rand_num_y:rand_num_y+crop_size,:]
        nm = Image.fromarray(nm)
        nc = Image.fromarray(nc)
        return nm,nc

class DatasetFromImage(data.Dataset):
    def __init__(self):
        super(DatasetFromImage, self).__init__()
        # root = '../../data/dl-sim/%s/Training_Testing_%s/'%(dataname,dataname)
        # [[path,title,image],....]
        self.HR_list = []
        self.LR_list = []
        print('init database')
        datanames = ['adhesion', 'factin', 'microtubule', 'mitochondria']
        for dataname in datanames:
            if dataname == 'microtubule' and cfg.datamode == 'LE':
                title_str = 'LE'
            else:
                title_str = 'HE'  
                
            root = '%s/%s/Training_Testing_%s/'%(cfg.data_path,dataname,dataname)
            if cfg.datamode == 'HE':
                MHR,MLR = ['%s/HER'%root,'%s/HE_X2'%root]
            elif cfg.datamode == 'LE':
                MHR,MLR = ['%s/HER'%root,'%s/LE_X2'%root]
            dataset_HR_list = os.listdir(MHR) # sample1.tif
            dataset_LR_list = os.listdir(MLR) # sample1/HE_00.tif
            # self.crop_size = 256
            for im_HR in dataset_HR_list:
                self.HR_list.append([MHR,MLR,title_str,im_HR])
            for im_LR in dataset_LR_list:
                self.LR_list.append([MHR,MLR,title_str,im_LR])
        print('dataset ok')        
             
        self.data_augment = cfg.data_augment   

    def __getitem__(self, index):
        base_HR, base_LR, title_str,im_HR = self.HR_list[index]
        HR = io.imread(os.path.join(base_HR,im_HR))
        # random choose a LR
        rand_index = np.random.randint(0,15)
        if rand_index < 10:
            local_LR_path = os.path.join(im_HR[:-4],'%s_0%d.tif'%(title_str,rand_index))
        else:
            local_LR_path = os.path.join(im_HR[:-4],'%s_%d.tif'%(title_str,rand_index))
        LR = io.imread(os.path.join(base_LR,local_LR_path))

        if self.data_augment:
            aug_num = np.random.randint(0, 8)
            HR = data_augment(HR,aug_num) 
            LR = data_augment(LR,aug_num)   

        THR = torch.from_numpy(HR.astype(np.float32)/cfg.max_hr).view(1,256,256)
        TLR = torch.from_numpy(LR.astype(np.float32)/cfg.max_lr).view(1,256,256)
        # print(LR.shape)
        return TLR, THR
        
    def __len__(self):
        return len(self.HR_list)

def test():
    file_path = "./"
    dfi = DatasetFromImage(file_path)
