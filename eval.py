import utils
import torch 
import os
import config
from utils import test
cfg = config.Config()

def run_test(testing_dataset,model,silent=False):
    rmse,ssim = utils.test.run_test(testing_dataset,model,cfg.save_path,silent=silent)
    return rmse,ssim

def run_test15(testing_dataset,model,silent=False):
    rmse,ssim = utils.test.run_test_15(testing_dataset,model,cfg.save_path,silent=silent)
    return rmse,ssim

def select_best(testing_dataset, nmodels = 30):
    best_rmse = 10000000.0
    for i in range(1,nmodels+1):
        path = os.path.join(cfg.ckpt_path,'model_epoch_%d.pth'%i)
        model = torch.load(path)["model"]
        rmse,ssim = run_test(testing_dataset,model,silent = True)
        print('RMSE =',rmse, 'model=',i)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = i
        print('Best RMSE =',best_rmse, 'best model=',best_model, 'SSIM = ',ssim)

def test_best(testing_dataset):
    path = os.path.join(cfg.ckpt_path,'model_epoch_best.pth')
    model = torch.load(path)["model"]
    rmse,ssim = run_test(testing_dataset,model,silent = False)
    print('Best RMSE =',rmse, 'SSIM = ',ssim)

def test_best_15(testing_dataset):
    path = os.path.join(cfg.ckpt_path,'model_epoch_best.pth')
    model = torch.load(path)["model"]
    rmse,ssim = test.run_test_15(testing_dataset,model,silent = False)
    print('Best RMSE =',rmse, '\n SSIM = ',ssim)

if __name__ == "__main__":
    # path = os.path.join(cfg.ckpt_path,'model_epoch_60.pth')
    # model = torch.load(path)["model"]
    dataname = cfg.dataname
    print(dataname)
    testing_dataset = utils.test.testing_dataset_preloader(dataname)
    # select_best(testing_dataset,30)
    test_best(testing_dataset)

    