import sys
import os
import datetime
import shutil
import torch

def make_print_to_file(path='./'):
    '''
    保存实验结果输出
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass

    if not os.path.exists(path):
        os.makedirs(path)
    fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

def save_experiment():
    '''
    保存当前实验内容
    '''
    root_path = './experiments'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    code_path = os.path.join(root_path,t)
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    copy_files('./',code_path)
    print('code copied to ',code_path)

def copy_files(source, target):
    '''
    复制文件
    '''
    files = os.listdir(source)
    for f in files:
        if f[-3:] == '.py' or f[-3:] == '.sh':
            print(f)
            shutil.copy(source+f, target)

def save_checkpoint(model, epoch, folder,name=None):
    '''
    保存模型
    '''
    model_folder = folder
    if name==None:
        model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    else:
        model_out_path = model_folder + "model_epoch_{}.pth".format(name)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))