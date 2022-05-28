from babel.plural import to_python
from skimage import io,transform,measure
import numpy as np
import os
import config
cfg = config.Config()
import torch
import multiprocessing
from tqdm import tqdm
from utils.misc import RMSE,map_im_range_max,calc_ssim
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import cv2

def testing_dataset_preloader(dataname):
    if cfg.dataset_name == 'W2S':
        testing_dataset = w2s_preloader(dataname)
    elif cfg.dataset_name == 'DL-SIM' and cfg.sample_large == False:
        testing_dataset = dlsim_preloader(dataname)
    elif cfg.dataset_name == 'DL-SIM' and cfg.sample_large == True:
        print('DL-SIM Large Sample')
        testing_dataset = dlsim_large_preloader(dataname)
    elif cfg.dataset_name == 'DL-SR':
        testing_dataset = dlsr_preloader(dataname)
    elif cfg.dataset_name == 'DL-SR2':
        testing_dataset = dlsr2_preloader(dataname)
    return testing_dataset

def dlsr_load_function(path_vgt,path_vwf,HR):
    iHR = io.imread(os.path.join(path_vgt,HR))    
    iLR = io.imread(os.path.join(path_vwf,HR))  
    if cfg.input_small:
        TLR = torch.from_numpy(iLR.astype(np.float32)/cfg.max_lr).view(1,1,128,128)
    else:
        iLR = transform.resize(iLR/65535.0,[256,256],order=3)*65535.0
        TLR = torch.from_numpy(iLR.astype(np.float32)/cfg.max_lr).view(1,1,256,256)
    res = [iHR,TLR,HR,1]
    return res

def dlsr2_load_function(path_vgt,path_frame,test_im):
    im_hr = io.imread(os.path.join(path_vgt, test_im+'.tif'))
    testing_dataset = []
    for i in range(9):
        tp = '%d.tif'%(i+1)
        im_lr = io.imread(os.path.join(path_frame,test_im,tp))
        im_lr = transform.resize(im_lr/65535,[256,256],order=3)*65535
        im_lr_t = torch.from_numpy(im_lr.astype(np.float32)/cfg.max_lr).view(1,1,256,256).cuda()
        testing_dataset.append([im_hr,im_lr_t,test_im,i])       

# def w2s_load_function(image_root,image):
#     import dataset
#     WF = np.load(os.path.join(
#         image_root, image,'wf_channel0.npy'))   # 400x512x512
#     nWF = np.zeros([512,512])
#     for i in range(cfg.nframes):
#         nWF += WF[i,:,:]
#     nWF = nWF/cfg.nframes        
#     SIM_gt = np.load(os.path.join(image_root, image,'sim_channel0.npy'))   #1024x1024
#     TLR,THR = dataset.random_crop(nWF,SIM_gt,crop_size=1024,fixed_pos=[0,0])
#     im_hr = THR.numpy()[0]
#     im_lr_t = TLR.view(1,1,1024,1024)
#     res = [im_hr,im_lr_t,image,1]
#     return res

def w2s_load_function(base_wf,base_gt,image):
    wf = io.imread(os.path.join(base_wf,image))
    gt = io.imread(os.path.join(base_gt,image))
    im_hr = gt
    wf = torch.from_numpy(wf.astype(np.float32)/cfg.max_lr)
    im_lr_t = wf.view(1,1,256,256)
    res = [im_hr,im_lr_t,image,1]
    return res


def dlsr_preloader(dataname):
    import dataset
    base = cfg.data_path
    image_root = os.path.join(base,cfg.dataname)
    path_vgt = os.path.join(image_root,'validate_gt')
    path_vwf = os.path.join(image_root,'validate_wf')
    HR_list = os.listdir(path_vgt)
    LR_list = os.listdir(path_vwf)
    testing_dataset = []  
    pool = multiprocessing.Pool(192)
    result =[]
    print('preloading testing images')  
    for HR in tqdm(HR_list):
        # res = dlsr_load_function(path_vgt,path_vwf,HR)
        # testing_dataset.append(res)
        result.append(pool.apply_async(
            func=dlsr_load_function, args=(path_vgt,path_vwf,HR)))        
    for res in result:
        testing_dataset.append(res.get())
    return testing_dataset     

def dlsr2_preloader(dataname): 
    base = cfg.data_path
    image_root = os.path.join(base,cfg.dataname)
    path_vgt = os.path.join(image_root,'validate_gt')
    path_frame = os.path.join(image_root,'validate')
    HR_list = os.listdir(path_vgt)
    LR_list = os.listdir(path_frame)   
    testing_dataset = []  
    pool = multiprocessing.Pool(192)
    result =[]
    print('preloading testing images')   
    for test_im in tqdm(LR_list):
        result.append(pool.apply_async(
            func=dlsr_load_function, args=(path_vgt,path_frame,test_im)))        
    for res in result:
        testing_dataset.append(res.get())  
    return testing_dataset


def w2s_preloader(dataname):
    testing_dataset = []  
    print('preloading testing images')  
    base_gt = os.path.join(cfg.data_path,'test_gt_level1')
    base_wf = os.path.join(cfg.data_path,'test_wf_level1')
    HRLR_dirs =  os.listdir(base_gt)
    HR_list = []
    for idir in HRLR_dirs:
        images = os.listdir(os.path.join(base_gt, idir))
        for image in images:
            HR_list.append(os.path.join(idir, image))

    # pool = multiprocessing.Pool(1960)
    # result =[]
    for image in tqdm(HR_list):
        res = w2s_load_function(base_wf,base_gt,image)
        testing_dataset.append(res)
    #     result.append(pool.apply_async(func=w2s_load_function, args=(base_wf,base_gt,image)))
    # for res in result:
    #     testing_dataset.append(res.get())
    return testing_dataset

def dlsim_preloader(dataname):
    if cfg.input_small == False:
        if dataname == 'microtubule' and cfg.datamode == 'LE':   # fix dataset name issue
            title_str = 'LE'
        else:
            title_str = 'HE'  
    else:
        if cfg.datamode == 'LE':  
            title_str = 'LE'
        else:
            title_str = 'HE'          
    # data_root = '../../data/dl-sim/%s/Training_Testing_%s/'%(dataname,dataname)
    data_root = '%s/%s/Training_Testing_%s/'%(cfg.data_path,dataname,dataname)
    test_path = os.path.join(data_root,'testing_HER')
    test_images = os.listdir(test_path)
    ntest = len(test_images)  # test all images    
    testing_dataset = []  
    print('preloading testing images')   
    for test_im in tqdm(test_images[:ntest]):
        im_hr = io.imread(os.path.join(test_path, test_im))
        if cfg.nframes == 1:
            if cfg.wf == False:
                if cfg.input_small == False:
                    for i in range(15):
                        if i < 10:
                            tp = '%s_0%d.tif'%(title_str,i)
                        else:
                            tp = '%s_%d.tif'%(title_str,i)
                        if cfg.datamode == 'HE':
                            im_lr = io.imread(os.path.join(data_root+'/testing_HE_X2/'+test_im[:-4],tp))
                        elif cfg.datamode == 'LE':
                            im_lr = io.imread(os.path.join(data_root+'/testing_LE_X2/'+test_im[:-4],tp))
                        im_lr_t = torch.from_numpy(im_lr.astype(np.float32)/cfg.max_lr).view(1,1,256,256).cuda()
                        testing_dataset.append([im_hr,im_lr_t,test_im,i])
                else:
                    for i in range(1,16):
                        tp = '%s_%d.tif'%(title_str,i)
                        if cfg.datamode == 'HE':
                            im_lr = io.imread(os.path.join(data_root+'/testing_HE/'+test_im[:-4],tp))
                        elif cfg.datamode == 'LE':
                            im_lr = io.imread(os.path.join(data_root+'/testing_LE/'+test_im[:-4],tp))
                        im_lr_t = torch.from_numpy(im_lr.astype(np.float32)/cfg.max_lr).view(1,1,128,128).cuda()
                        testing_dataset.append([im_hr,im_lr_t,test_im,i])                    
            else:
                im_lr_arr = np.zeros([1,256,256])
                for i in range(15):
                    if i < 10:
                        tp = '%s_0%d.tif'%(title_str,i)
                    else:
                        tp = '%s_%d.tif'%(title_str,i)
                    if cfg.datamode == 'HE':
                        im_lr = io.imread(os.path.join(data_root+'/testing_HE_X2/'+test_im[:-4],tp))
                    elif cfg.datamode == 'LE':
                        im_lr = io.imread(os.path.join(data_root+'/testing_LE_X2/'+test_im[:-4],tp))
                    im_lr_arr[0,:,:] += im_lr
                    im_lr_arr = im_lr_arr/15
                im_lr_t = torch.from_numpy(im_lr_arr.astype(np.float32)/cfg.max_lr).view(1,1,256,256).cuda()
                testing_dataset.append([im_hr,im_lr_t,test_im,i])
        elif cfg.nframes == 3:
            im_lr_arr = np.zeros([3,256,256])
            for i in range(3):
                ii=i*5
                # if i < 10:
                #     tp = '%s_0%d.tif'%(title_str,i)
                # else:
                #     tp = '%s_%d.tif'%(title_str,i)
                if ii <= 9:
                    tp = "%s_0"%title_str+str(ii)+".tif"
                else:
                    tp = "%s_"%title_str+str(ii)+".tif"
                if cfg.datamode == 'HE':
                    im_lr = io.imread(os.path.join(data_root+'/testing_HE_X2/'+test_im[:-4],tp))
                elif cfg.datamode == 'LE':
                    im_lr = io.imread(os.path.join(data_root+'/testing_LE_X2/'+test_im[:-4],tp))
                im_lr_arr[i,:,:] = im_lr
            im_lr_t = torch.from_numpy(im_lr_arr.astype(np.float32)/cfg.max_lr).view(1,3,256,256).cuda()
            testing_dataset.append([im_hr,im_lr_t,test_im,i])
        elif cfg.nframes == 15:
            im_lr_arr = np.zeros([15,256,256])
            for i in range(15):
                if i < 10:
                    tp = '%s_0%d.tif'%(title_str,i)
                else:
                    tp = '%s_%d.tif'%(title_str,i)
                if cfg.datamode == 'HE':
                    im_lr = io.imread(os.path.join(data_root+'/testing_HE_X2/'+test_im[:-4],tp))
                elif cfg.datamode == 'LE':
                    im_lr = io.imread(os.path.join(data_root+'/testing_LE_X2/'+test_im[:-4],tp))
                im_lr_arr[i,:,:] = im_lr
            im_lr_t = torch.from_numpy(im_lr_arr.astype(np.float32)/cfg.max_lr).view(1,15,256,256).cuda()
            testing_dataset.append([im_hr,im_lr_t,test_im,i])
    return testing_dataset


def dlsim_large_preloader(dataname):
    if cfg.datamode == 'LE':   # fix dataset name issue
        title_str = 'LE'
    else:
        title_str = 'HE'  
    data_root = '%s/%s/Samples/'%(cfg.data_path,dataname)
    test_path = os.path.join(data_root,'HER')
    test_images = os.listdir(test_path)
    ntest = len(test_images)  # test all images    
    testing_dataset = []  
    print('preloading testing images')   
    for test_im in tqdm(test_images[:ntest]):
        if test_im.endswith('.tif'):
            im_hr = io.imread(os.path.join(test_path, test_im))
            if cfg.nframes == 1:
                for i in range(1,16,1):
                    if i < 10:
                        tp = '%s_%d.tif'%(title_str,i)
                    else:
                        tp = '%s_%d.tif'%(title_str,i)
                    if cfg.datamode == 'HE':
                        im_lr = io.imread(os.path.join(data_root+'/HE/'+test_im[:-4],tp))
                        if not cfg.input_small:
                            im_lr = transform.resize(im_lr/65535,[1024,1024],order=3)*65535
                    elif cfg.datamode == 'LE':
                        im_lr = io.imread(os.path.join(data_root+'/LE/'+test_im[:-4],tp))
                        if not cfg.input_small:
                            im_lr = transform.resize(im_lr/65535,[1024,1024],order=3)*65535
                    if not cfg.input_small:
                        im_lr_t = torch.from_numpy(im_lr.astype(np.float32)/cfg.max_lr).view(1,1,1024,1024).cuda()
                    else:
                        im_lr_t = torch.from_numpy(im_lr.astype(np.float32)/cfg.max_lr).view(1,1,512,512).cuda()
                    testing_dataset.append([im_hr,im_lr_t,test_im,i])
            elif cfg.nframes == 15:
                im_lr_arr = np.zeros([15,256,256])
                for i in range(15):
                    if i < 10:
                        tp = '%s_0%d.tif'%(title_str,i)
                    else:
                        tp = '%s_%d.tif'%(title_str,i)
                    if cfg.datamode == 'HE':
                        im_lr = io.imread(os.path.join(data_root+'/testing_HE_X2/'+test_im[:-4],tp))
                    elif cfg.datamode == 'LE':
                        im_lr = io.imread(os.path.join(data_root+'/testing_LE_X2/'+test_im[:-4],tp))
                    im_lr_arr[i,:,:] = im_lr
                im_lr_t = torch.from_numpy(im_lr_arr.astype(np.float32)/cfg.max_lr).view(1,15,256,256).cuda()
                testing_dataset.append([im_hr,im_lr_t,test_im,i])
    return testing_dataset

def run_test(testing_dataset,model,save_path=None,silent=True):
    model.eval()
    m_rmse_list = []   
    m_ssim_list = []
    if not silent:
        testing_dataset = tqdm(testing_dataset)
    for test_data in testing_dataset:
        rmse_list = []  
        ssim_list = []
        im_hr,im_lr_t,test_im,i = test_data
        with torch.no_grad():
            GHR = model(im_lr_t.cuda())
        im_GHR = GHR.cpu().data[0].numpy() #BCHW->CHW
        im_GHR = np.clip(im_GHR, 0., 1.)[0]  # HW
        # print(im_GHR.shape)
        # print(im_hr.shape)
        rmse = RMSE(im_GHR,im_hr.astype(np.float32)/cfg.max_hr)*cfg.max_range
        ssim = calc_ssim(im_GHR*255.0,(im_hr.astype(np.float32)/cfg.max_hr)*255.0)
        # ms-ssim
        # ssim = ms_ssim(
        #     torch.from_numpy(im_GHR).view(1,1,256,256), 
        #     torch.from_numpy(im_hr.astype(np.float32)/cfg.max_hr).view(1,1,256,256), data_range=1, size_average=False )
        rmse_list.append(rmse)
        ssim_list.append(ssim)
        if save_path != None:
            # im_GHR = map_im_range(im_GHR,data_range1=1,data_range2=cfg.max_range_rgb)
            im_GHR = map_im_range_max(im_GHR)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # im_GHR =  cv2.applyColorMap(im_GHR, cv2.COLORMAP_HOT)    #不用color map看下效果
            if cfg.dataname == 'W2S':
                cv2.imwrite(os.path.join(save_path,test_im+'-'+str(i)+cfg.save_image_ext),im_GHR)
            else:
                cv2.imwrite(os.path.join(save_path,test_im[:-4]+'-'+str(i)+cfg.save_image_ext),im_GHR)

        m_rmse = np.mean(rmse_list)
        m_ssim = np.mean(ssim_list)
        m_rmse_list.append(m_rmse)
        m_ssim_list.append(m_ssim)
    return np.mean(m_rmse_list),np.mean(m_ssim_list)

def run_test_15(testing_dataset,model,save_path=None,silent=True):
    model.eval()

    rmse_1 = []
    rmse_2 = []
    rmse_3 = []
    rmse_4 = []
    rmse_5 = []
    rmse_6 = []
    rmse_7 = []
    rmse_8 = []
    rmse_9 = []
    rmse_10 = []
    rmse_11 = []
    rmse_12 = []
    rmse_13 = []
    rmse_14 = []
    rmse_15 = []
    ssim_1 = []
    ssim_2 = []
    ssim_3 = []
    ssim_4 = []
    ssim_5 = []
    ssim_6 = []
    ssim_7 = []
    ssim_8 = []
    ssim_9 = []
    ssim_10 = []
    ssim_11 = []
    ssim_12 = []
    ssim_13 = []
    ssim_14 = []
    ssim_15 = []
    m_rmse_list = [rmse_1,rmse_2,rmse_3,rmse_4,rmse_5,rmse_6,rmse_7,rmse_8,rmse_9,rmse_10,rmse_11,rmse_12,rmse_13,rmse_14,rmse_15]   
    m_ssim_list = [ssim_1,ssim_2,ssim_3,ssim_4,ssim_5,ssim_6,ssim_7,ssim_8,ssim_9,ssim_10,ssim_11,ssim_12,ssim_13,ssim_14,ssim_15]
    if not silent:
        testing_dataset = tqdm(testing_dataset)
    for test_data in testing_dataset:

        im_hr,im_lr_t,test_im,i = test_data
        with torch.no_grad():
            GHR = model(im_lr_t.cuda())
        im_GHR = GHR.cpu().data[0].numpy() #BCHW->CHW
        im_GHR = np.clip(im_GHR, 0., 1.)[0]  # HW

        rmse = RMSE(im_GHR,im_hr.astype(np.float32)/cfg.max_hr)*cfg.max_range
        ssim = calc_ssim(im_GHR*255.0,(im_hr.astype(np.float32)/cfg.max_hr)*255.0)
        m_rmse_list[i].append(rmse)
        m_ssim_list[i].append(ssim)
        if save_path != None:
            # im_GHR = map_im_range(im_GHR,data_range1=1,data_range2=cfg.max_range_rgb)
            im_GHR = map_im_range_max(im_GHR)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # im_GHR =  cv2.applyColorMap(im_GHR, cv2.COLORMAP_HOT)    #不用color map看下效果
            if cfg.dataname == 'W2S':
                cv2.imwrite(os.path.join(save_path,test_im+'-'+str(i)+cfg.save_image_ext),im_GHR)
            else:
                cv2.imwrite(os.path.join(save_path,test_im[:-4]+'-'+str(i)+cfg.save_image_ext),im_GHR)

    new_rmse = []
    new_ssim = []
    for i in range(len(m_rmse_list)):
        new_rmse.append(np.mean(m_rmse_list[i]))
        new_ssim.append(np.mean(m_ssim_list[i]))
    return new_rmse,new_ssim