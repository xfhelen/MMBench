import os
import numpy as np
import medpy.io as medio
join=os.path.join
#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict
import torch.profiler
import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

import mmformer
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn

src_path = 'datasets/medical-segmentation/Training'
tar_path = './BRATS2018_Training_none_npy'

class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler
    def __iter__(self):
        while True:
            yield from iter(self.sampler)
class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
    def __len__(self):
        return len(self.batch_sampler.sampler)
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print ('#' * 100)
        ecart = int((128-(xmax-xmin))/2)
        xmax = xmax+ecart+1
        xmin = xmin-ecart
    if xmin < 0:
        xmax-=xmin
        xmin=0
    return xmin, xmax

def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_dim, y_dim, z_dim = tuple(vol.shape)
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max

def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x

    return vol





masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']


parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--options', default="normal", type=str, help='mode')
parser.add_argument('--dataname', default='BRATS2018', type=str)
parser.add_argument('--savepath', default='./output', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1024, type=int)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
options = args.options


def main():
    if options == 'encoder' or options == 'normal': 


        HGG_list = os.listdir(join(src_path, 'HGG'))
        HGG_list = ['HGG/'+x for x in HGG_list]
        LGG_list = os.listdir(join(src_path, 'LGG'))
        LGG_list = ['LGG/'+x for x in LGG_list]
        name_list = HGG_list + LGG_list
        
        if not os.path.exists(os.path.join(tar_path, 'vol')):
            os.makedirs(os.path.join(tar_path, 'vol'))

        if not os.path.exists(os.path.join(tar_path, 'seg')):
            os.makedirs(os.path.join(tar_path, 'seg'))
        i = 0
        for file_name in name_list:
            print (file_name)
            if 'HGG' in file_name:
                HLG = 'HGG_'
            else:
                HLG = 'LGG_'
            case_id = file_name.split('/')[-1]
            flair, flair_header = medio.load(os.path.join(src_path, file_name, case_id+'_flair.nii.gz'))
            t1ce, t1ce_header = medio.load(os.path.join(src_path, file_name, case_id+'_t1ce.nii.gz'))
            t1, t1_header = medio.load(os.path.join(src_path, file_name, case_id+'_t1.nii.gz'))
            t2, t2_header = medio.load(os.path.join(src_path, file_name, case_id+'_t2.nii.gz'))

            vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
            x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
            vol1 = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])
            vol1 = vol1.transpose(1,2,3,0)
            print (vol1.shape)

            seg, seg_header = medio.load(os.path.join(src_path, file_name, case_id+'_seg.nii.gz'))
            seg = seg.astype(np.uint8)
            seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
            seg1[seg1==4]=3

            np.save(os.path.join(tar_path, 'vol', HLG+case_id+'_vol.npy'), vol1)
            np.save(os.path.join(tar_path, 'seg', HLG+case_id+'_seg.npy'), seg1)

            if  i == 2 : 
                break
            i = i + 1


    ckpts = args.savepath
    os.makedirs(ckpts, exist_ok=True)

    
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model = mmformer.Model(num_cls=num_cls)
    model = model.cuda()


    ##########Setting data
    if args.dataname in ['BRATS2020', 'BRATS2015']:
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train3.txt'
        test_file = 'test3.txt'

    datapath = './BRATS2018_Training_none_npy'
    if options == 'encoder' or options == 'normal' :
        train_set = Brats_loadall_nii(transforms=args.train_transforms, root=datapath, num_cls=num_cls, train_file=train_file)
        test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=datapath, test_file=test_file)
        train_loader = MultiEpochsDataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=init_fn)
        iter_per_epoch = len(train_loader)
        train_iter = iter(train_loader)
    for epoch in range(args.num_epochs):
        if options == 'encoder' or options == 'normal' :
            for i in range(iter_per_epoch):
                step = (i+1) + epoch*iter_per_epoch
                ###Data load
                try:
                    data = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    data = next(train_iter)
                x, target, mask = data[:3]
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                model.is_training = False
                model(options, x, mask)
                break
        else :
            for i in range(2):
                x = torch.ones([1, 4, 128, 128, 128]).to('cuda')
                mask = torch.ones((1,4),dtype=torch.bool).to('cuda')
                model(options, x,mask)
            
        break


    print('start inference')
    option = "normal"
    with torch.no_grad():
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './log/seg_agg'),
            record_shapes=True,
            with_stack=True)
        with prof as p: 

            train_set = Brats_loadall_nii(transforms=args.train_transforms, root=datapath, num_cls=num_cls, train_file=train_file)
            train_loader = MultiEpochsDataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle=True,
                worker_init_fn=init_fn)

            for i in range(iter_per_epoch):
                step = (i+1) + epoch*iter_per_epoch
                ###Data load
                try:
                    data = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    data = next(train_iter)
                x, target, mask = data[:3]
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                model.is_training = False
                model(option, x, mask)
                p.step()

if __name__ == '__main__':
    main()

