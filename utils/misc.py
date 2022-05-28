import torch
import cv2
import numpy as np
import os    
from tqdm import tqdm
import colour
from skimage import io
import math
from PIL import Image

def calc_psnr_from_folder(src_path, dst_path):
    '''
    计算PSNR
    '''
    src_image_name = os.listdir(src_path)
    dst_image_name = os.listdir(dst_path)
    image_label = ['_'.join(i.split("_")[:-1]) for i in src_image_name]
    num_image = len(src_image_name)
    psnr = 0
    for ii, label in tqdm(enumerate(image_label)):
        src = os.path.join(src_path, "{}_source.png".format(label))
        dst = os.path.join(dst_path, "{}_target.png".format(label))
        src_image = default_loader(src)
        dst_image = default_loader(dst)

        single_psnr = colour.utilities.metric_psnr(src_image, dst_image, 255)
        psnr += single_psnr

    psnr /= num_image
    return psnr

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''
    计算 结构相似度 SSIM
    calculate SSIM the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_mean(path):
    '''
    计算 RGB 均值
    '''
    images = os.listdir(path)
    lr = []
    lg = []
    lb = []
    for img in tqdm(images):
        full_path = os.path.join(path, img)
        im = io.imread(full_path)
        im_r = np.mean(im[:,:,0])
        im_g = np.mean(im[:,:,1])
        im_b = np.mean(im[:,:,2])
        lr.append(im_r)
        lg.append(im_g)
        lb.append(im_b)
    mean_r = np.mean(lr)
    mean_g = np.mean(lg)
    mean_b = np.mean(lb)
    return mean_r, mean_g, mean_b        
            
def RMSE(pred, gt, shave_border=2):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    return rmse

def PSNR(pred, gt, shave_border=0, data_range=65535.0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(data_range / rmse)

def map_im_range(img,data_range1=1,data_range2=255.0):
    new_img = data_range2*(img/data_range1)
    return new_img.astype(np.uint8)

def map_im_range_max(img):
    max_data = np.amax(img)
    img = img/max_data
    img = img*255.0
    return img.astype(np.uint8)


def tensor2im(input_image, imtype=np.uint8):
    '''
    # Converts a Tensor into an image array (numpy)
    # |imtype|: the desired type of the converted numpy array
    '''
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.detach()
    else:
        return input_image
    image_numpy = image_tensor.cpu().numpy().astype(np.float32)
    image_numpy = image_numpy*255.
    image_numpy = np.clip(image_numpy, 0., 255.)
    # image_numpy = (image_numpy + 1.0) / 2.0
    return image_numpy.astype(imtype)

def save_single_image(img, img_path):
    '''
    CV2 保存单张图片
    '''
    img = np.transpose(img, (1, 2, 0))  # C,H,W --> H,W,C
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # CV2的颜色是BGR
    img = img * 255
    cv2.imwrite(img_path, img)
    return img

def default_loader(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
    return region

class data_prefetcher():
    '''
    NVIDIA的dataloader
    '''
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()  
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

def single_to_dp_model(model,saved_state):
    '''
    单卡模型转换多卡
    '''
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in saved_state.items():
        namekey = 'module.'+k
        new_state_dict[namekey] = v
    model.load_state_dict(new_state_dict)
    return model    