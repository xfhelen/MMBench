import argparse
import json
import os
from tqdm import tqdm


import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data, lidar_bev_cam_correspondences

import pathlib
import datetime
from torch.distributed.elastic.multiprocessing.errors import record
import random
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp

from diskcache import Cache    
@record   
def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='transfuser', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=41, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for one GPU. When training with multiple GPUs the effective batch size will be batch_size*num_gpus')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--load_file', type=str, default=None, help='ckpt to load.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start with. Useful when continuing trainings via load_file.')
    parser.add_argument('--setting', type=str, default='all', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')
    parser.add_argument('--root_dir', type=str, default=r'/mnt/qb/geiger/kchitta31/datasets/carla/pami_v1_dataset_23_11', help='Root directory of your training data')
    parser.add_argument('--schedule', type=int, default=1,
                        help='Whether to train with a learning rate schedule. 1 = True')
    parser.add_argument('--schedule_reduce_epoch_01', type=int, default=30,
                        help='Epoch at which to reduce the lr by a factor of 10 the first time. Only used with --schedule 1')
    parser.add_argument('--schedule_reduce_epoch_02', type=int, default=40,
                        help='Epoch at which to reduce the lr by a factor of 10 the second time. Only used with --schedule 1')
    parser.add_argument('--backbone', type=str, default='transFuser',
                        help='Which Fusion backbone to use. Options: transFuser, late_fusion, latentTF, geometric_fusion')
    parser.add_argument('--image_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the image branch. efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--lidar_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the lidar branch. Tested: efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--use_velocity', type=int, default=0,
                        help='Whether to use the velocity input. Currently only works with the TransFuser backbone. Expected values are 0:False, 1:True')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers used in the transfuser')
    parser.add_argument('--wp_only', type=int, default=0,
                        help='Valid values are 0, 1. 1 = using only the wp loss; 0= using all losses')
    parser.add_argument('--use_target_point_image', type=int, default=1,
                        help='Valid values are 0, 1. 1 = using target point in the LiDAR0; 0 = dont do it')
    parser.add_argument('--use_point_pillars', type=int, default=0,
                        help='Whether to use the point_pillar lidar encoder instead of voxelization. 0:False, 1:True')
    parser.add_argument('--parallel_training', type=int, default=1,
                        help='If this is true/1 you need to launch the train.py script with CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=123456780 --rdzv_backend=c10d train.py '
                             ' the code will be parallelized across GPUs. If set to false/0, you launch the script with python train.py and only 1 GPU will be used.')
    parser.add_argument('--val_every', type=int, default=5, help='At which epoch frequency to validate.')
    parser.add_argument('--no_bev_loss', type=int, default=0, help='If set to true the BEV loss will not be trained. 0: Train normally, 1: set training weight for BEV to 0')
    parser.add_argument('--sync_batch_norm', type=int, default=0, help='0: Compute batch norm for each GPU independently, 1: Synchronize Batch norms accross GPUs. Only use with --parallel_training 1')
    parser.add_argument('--zero_redundancy_optimizer', type=int, default=0, help='0: Normal AdamW Optimizer, 1: Use Zero Reduncdancy Optimizer to reduce memory footprint. Only use with --parallel_training 1')
    parser.add_argument('--use_disk_cache', type=int, default=0, help='0: Do not cache the dataset 1: Cache the dataset on the disk pointed to by the SCRATCH enironment variable. Useful if the dataset is stored on slow HDDs and can be temporarily stored on faster SSD storage.')


    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)
    parallel = bool(args.parallel_training)

    if(bool(args.use_disk_cache) == True):
        if (parallel == True):
            # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
            # During training we cache the dataset on the fast storage of the local compute nodes.
            # Adapt to your cluster setup as needed.
            # Important initialize the parallel threads from torch run to the same folder (so they can share the cache).
            tmp_folder = str(os.environ.get('SCRATCH'))
            print("Tmp folder for dataset cache: ", tmp_folder)
            tmp_folder = tmp_folder + "/dataset_cache"
            # We use a local diskcache to cache the dataset on the faster SSD drives on our cluster.
            shared_dict = Cache(directory=tmp_folder ,size_limit=int(768 * 1024 ** 3))
        else:
            shared_dict = Cache(size_limit=int(768 * 1024 ** 3))
    else:
        shared_dict = None

    # Use torchrun for starting because it has proper error handling. Local rank will be set automatically
    if(parallel == True): #Non distributed works better with my local debugger
        rank       = int(os.environ["RANK"]) #Rank accross all processes
        local_rank = int(os.environ["LOCAL_RANK"]) # Rank on Node
        world_size = int(os.environ['WORLD_SIZE']) # Number of processes
        print(f"RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}")

        device = torch.device('cuda:{}'.format(local_rank))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank) # Hide devices that are not used by this process

        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank,
                                             timeout=datetime.timedelta(minutes=15))

        torch.distributed.barrier(device_ids=[local_rank])
    else:
        rank       = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda:{}'.format(local_rank))

    torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = True # Wen want the highest performance

    # Configure config
    config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)
    config.use_target_point_image = bool(args.use_target_point_image)
    config.n_layer = args.n_layer
    config.use_point_pillars = bool(args.use_point_pillars)
    config.backbone = args.backbone
    if(bool(args.no_bev_loss)):
        index_bev = config.detailed_losses.index("loss_bev")
        config.detailed_losses_weights[index_bev] = 0.0

    # Create model and optimizers
    model = LidarCenterNet(config, device, args.backbone, args.image_architecture, args.lidar_architecture, bool(args.use_velocity))

    if (parallel == True):
        # Synchronizing the Batch Norms increases the Batch size with which they are compute by *num_gpus
        if(bool(args.sync_batch_norm) == True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    model.cuda(device=device)

    if ((bool(args.zero_redundancy_optimizer) == True) and (parallel == True)):

        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=optim.AdamW, lr=args.lr) # Saves GPU memory during DDP training
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr) # For single GPU training


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('Total trainable parameters: ', params)

    # Data
    train_set = CARLA_Data(root=config.train_data, config=config, shared_dict=shared_dict)
    val_set   = CARLA_Data(root=config.val_data,   config=config, shared_dict=shared_dict)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())

    if(parallel == True):
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, num_replicas=world_size, rank=rank)
        sampler_val   = torch.utils.data.distributed.DistributedSampler(val_set,   shuffle=True, num_replicas=world_size, rank=rank)
        dataloader_train = DataLoader(train_set, sampler=sampler_train, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=8, pin_memory=True)
        dataloader_val   = DataLoader(val_set,   sampler=sampler_val,   batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=8, pin_memory=True)
    else:
      dataloader_train = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)
      dataloader_val   = DataLoader(val_set,   shuffle=True, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)


    print(len(dataloader_train))

    for batch in dataloader_train:
        #print(batch)  # 打印整个批次的内容

        keys = batch.keys()  # 获取字典中的键
        print(keys)  # 打印字典中的键列表

    # 遍历字典中的每个键，并打印对应的值
        for key in keys:
            value = batch[key]
            print(f"键 '{key}' 的tensor的shape: {value.shape}")
        
        break
         


# We need to seed the workers individually otherwise random processes in the dataloader return the same values across workers!
def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

main()

print("niumomo")