# -*- coding:utf-8 -*-
import argparse, os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from dataset import DatasetFromImage
from utils.files import save_experiment,make_print_to_file,save_checkpoint
from utils.misc import single_to_dp_model
from utils.test import run_test,testing_dataset_preloader,run_test_15
import time
from skimage import io
import config
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
warnings.filterwarnings("ignore")

cfg = config.Config()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    return cfg.init_learning_rate * (0.1 ** (epoch // cfg.lr_decay_step))

def train(training_data_loader, optimizer, model, criterions_list, epoch, best_rmse, testing_dataset):
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"]) 
    model.train()
    init_time = time.time()

    if cfg.use_amp:
        scaler = GradScaler()
    for iteration, batch in enumerate(training_data_loader, 1):
        optimizer.zero_grad()
        moire = batch[0].cuda()
        # print(moire.shape)
        clean = batch[1].cuda()

        if cfg.use_amp:
            with autocast():
                outputs = model(moire)
                loss = 0
                for criterion in criterions_list:
                    loss += criterion(outputs,clean)
                loss = loss / len(criterions_list)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  
        else:
            outputs = model(moire)
            loss = 0
            for criterion in criterions_list:
                loss += criterion(outputs,clean)
            loss = loss / len(criterions_list)
            loss.backward()
            optimizer.step()    

        show_iter = cfg.show
        if iteration%show_iter == 0:
            current_time = time.time()
            used_time = (current_time - init_time)/show_iter
            init_time = current_time
            rmse,ssim = run_test(testing_dataset, model)
            rmse_15,ssim_15 = run_test_15(testing_dataset, model)
            print("rmse_15 =",rmse_15,"ssim_15 =",ssim_15)
            if rmse<best_rmse:
                best_rmse = rmse
                save_checkpoint(model, epoch,folder=cfg.ckpt_path, name='best')
            print("===> Epoch[{}]({}/{}): Loss:{:.5f} Time used: {:.2f} /iter Test:{:.5f} Best:{:.5f} SSIM: {:.5f}".format(
                epoch, iteration, len(training_data_loader), loss.item(), used_time,rmse,best_rmse,ssim))
    return best_rmse

def main():
    global model, dataname
    make_print_to_file(path=cfg.ckpt_path)
    print("Start Training")
    if not torch.cuda.is_available():
        raise Exception("No GPU found !!! Please check your device and driver!!!")
    print("Random Seed: ", cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    cudnn.benchmark = True
    print("===> Loading datasets")
    dataname = cfg.dataname  
    
    train_set = DatasetFromImage(dataname)
    training_data_loader = DataLoader(
        dataset=train_set, num_workers=cfg.threads,batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

    testing_dataset = testing_dataset_preloader(dataname)

    print("===> Building model")
    model = cfg.model
    model=nn.DataParallel(model,device_ids=cfg.gpu_ids).cuda()

    criterions = cfg.criterions
    criterions_list = [] # pack criterions
    for criterion in criterions:
        criterions_list.append(criterion.cuda())    

    load_single_GPU_mode = False
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            cfg.start_epoch = checkpoint["epoch"] + 1
            saved_state = checkpoint["model"].state_dict()
            if load_single_GPU_mode:
                single_to_dp_model(model,saved_state)
            else: 
                model.load_state_dict(saved_state)
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))
    if cfg.pretrained:
        if os.path.isfile(cfg.pretrained):
            print("=> loading model '{}'".format(cfg.pretrained))
            weights = torch.load(cfg.pretrained)
            pretrained_dict = weights['model'].state_dict()
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
        else:
            print("=> no model found at '{}'".format(cfg.pretrained))

    print("===> Training")
    optimizer = optim.Adam(model.parameters(), lr=cfg.init_learning_rate)
    best_rmse = 1000000
    for epoch in range(cfg.start_epoch, cfg.total_epochs + 1):
        start = time.time()
        best_rmse = train(training_data_loader, optimizer, model, criterions_list, epoch, best_rmse, testing_dataset)
        elapsed = time.time() - start
        print("Time: %.2fs/Epoch"%elapsed)
        save_checkpoint(model, epoch, folder=cfg.ckpt_path)
    print("Training finished, Best RMSE: %.5f"%best_rmse)

if __name__ == "__main__":
    save_experiment()
    main()
