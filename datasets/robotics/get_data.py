"""Implements dataloaders for robotics data."""
import numpy as np
import os

from .utils import augment_val

from datasets.robotics.ToTensor import ToTensor
from datasets.robotics.ProcessForce import ProcessForce
from datasets.robotics.MultimodalManipulationDataset import MultimodalManipulationDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


def combine_modalitiesbuilder(unimodal, output):
    """Create a function data combines modalities given the type of input.

    Args:
        unimodal (str): Input type as a string. Can be 'force', 'proprio', 'image'. Defaults to using all modalities otherwise
        output (int): Index of output modality.
    """
    def _combine_modalities(data):
        if unimodal == "force":
            return [data['force'], data['action'], data[output]]
        if unimodal == "proprio":
            return [data['proprio'], data['action'], data[output]]
        if unimodal == "image":
            return [data['image'], data['depth'].transpose(0, 2).transpose(1, 2), data['action'], data[output]]
        return [
            data['image'],
            data['force'],
            data['proprio'],
            data['depth'].transpose(0, 2).transpose(1, 2),
            data['action'],
            data[output],
        ]
    return _combine_modalities


def get_data(device, configs, unimodal=None, output='contact_next'):
    # 得到所有以.h5结尾的文件名
    filename_list = []
    for file in os.listdir(configs['dataset']):
        if file.endswith(".h5"):
            filename_list.append(configs['dataset'] + file)

    val_filename_list = filename_list

    # 对训练集和验证集文件列表做一点处理
    val_filename_list1, _ = augment_val(val_filename_list, [])

    dataloaders = {}
    samplers = {}
    datasets = {}

    # 创建了采样器对象，用于验证集
    samplers["val"] = SubsetRandomSampler(range(len(val_filename_list1) * (configs['ep_length'] - 1)))

    datasets["val"] = MultimodalManipulationDataset(
        val_filename_list1,
        transform=transforms.Compose(
            [
                ProcessForce(32, "force", tanh=True),
                ProcessForce(32, "unpaired_force", tanh=True),
                ToTensor(device=device),
                combine_modalitiesbuilder(unimodal, output),
            ]
        ),
        episode_length=configs['ep_length'],
        training_type=configs['training_type'],
        action_dim=configs['action_dim'],
    )

    # 创建训练集和验证集的数据加载器对象
    dataloaders["val"] = DataLoader(
        datasets["val"],
        batch_size=configs['batch_size'],
        num_workers=configs['num_workers'],
        sampler=samplers["val"],
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloaders['val']
