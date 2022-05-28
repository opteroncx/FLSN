from skimage import measure,io
import os
import sys
sys.path.append('../')
from utils.misc import calc_ssim
import codecs
import numpy as np
result_file = codecs.open('ssim_result.csv','w', 'utf-8')

modes = ['LE','HE']
samples = ['adhesion','factin','microtubule','mitochondria']
methods = ['DFCAN','UNET','WMDN13']
gt_dir = '/test/sim/data/dl-sim-clean/'


for mode in modes:
    result_file.writelines('---------------------\n')
    result_file.write('adhesion,factin,microtubule,mitochondria\n')
    for sample in samples:
        result_file.write(sample+',')
        for method in methods:
            print('Method:%s'%method)
            image_path = './%s/65535-max-%s-%s-%s'%(mode,method,mode,sample)
            ssim_list = []
            images = os.listdir(image_path)
            for image in images:
                gt_path = gt_dir+'/'+sample+'/Training_Testing_%s/HER/image'%sample #改
                gt = io.imread(gt_path)
                sr = io.imread(os.path.join(image_path,image))
                ssim = calc_ssim(gt,sr)
                ssim_list.append(ssim)
            current_ssim = np.mean(ssim_list)
            result_file.write(str(current_ssim)+',')
        result_file.write('\n')

            
