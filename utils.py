import pytorch_lightning as pl
import logging
import torch 
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
from models import *

def get_key_avg(iters,key):
    queue = []

    for item in iters:
        queue.append(item[key])
    avg_queue = sum(queue)/len(queue)
    
    return avg_queue

def mkdir(path):
    
    if os.path.exists(path): 
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


def get_score(acc_list):

    mean = np.mean(acc_list)
    interval = 1.96*np.sqrt(np.var(acc_list)/len(acc_list))

    return mean,interval

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def pdist(x,y):
        '''
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        return dist

def torch_cov(m, rowvar=False):
    if m.size(0) == 1:
        return m
    if rowvar:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t() # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def lr_foo(epoch, start_epoch, gamma):      
    # warmup schedule
    lr_scale = 1.0
    if epoch >= start_epoch:

        lr_scale = lr_scale * gamma ** (epoch - start_epoch)
        
    return lr_scale


def get_logger(log_path_dict):

    filename = os.path.join(log_path_dict['save_dir'],log_path_dict['name'],log_path_dict['log_name'])

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",datefmt='%m/%d %I:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename,"w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def logger_show_config(logger:logging, params):

    logger.info('显示 yaml文件的所有超参数:')

    for k, v in params.items():
        logger.info(f'-----{k}-----')
        for k_i, v_i in params[k].items():
            logger.info('%s: %s' % (str(k_i),str(v_i)))

    logger.info('------------------------')

# get pre-resized 84x84 images for validation and test
def get_pre_folder(image_folder,transform_type):
    split = ['val','test']

    if transform_type == 0:
        transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224)])
    elif transform_type == 1:
        transform = transforms.Compose([transforms.Resize(84),
                                    transforms.CenterCrop(84)])

    for i in split:
        
        cls_list = os.listdir(os.path.join(image_folder,i))

        folder_name = i+'_pre_224'

        mkdir(os.path.join(image_folder,folder_name))

        for j in tqdm(cls_list):

            mkdir(os.path.join(image_folder,folder_name,j))

            img_list = os.listdir(os.path.join(image_folder,i,j))

            for img_name in img_list:
        
                img = Image.open(os.path.join(image_folder,i,j,img_name))
                img = img.convert('RGB')
                img = transform(img)
                img.save(os.path.join(image_folder,folder_name,j,img_name[:-3]+'png'))

