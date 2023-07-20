# Project:
#   VQA
# Description:
#   Loaders creation
# Author: 
#   Sergio Tascon-Morales

import numpy as np
from torch.utils.data import DataLoader
import collections
import torch
import random
from . import visual
from . import vqa, classif, nlp


def collater(batch):
    # function to collate several samples of a batch
    if torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif type(batch[0]).__module__ == np.__name__ and type(batch[0]).__name__ == 'ndarray':
        return torch.stack([torch.from_numpy(sample) for sample in batch], 0)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.tensor(batch) # * use DoubleTensor?
    elif isinstance(batch[0], dict):
        res = dict.fromkeys(batch[0].keys())
        for k in res.keys():
            res[k] = [s[k] for s in batch]
        return {k:collater(v) for k,v in res.items()}
    elif isinstance(batch[0], collections.Iterable):
        return torch.tensor(batch, dtype=torch.int) # ! integers because only for all answers it gets here and the indices are integers
    else:
        raise ValueError("Unknown type of samples in the batch. Add condition to collater function")


def collater_mainsub(batch):
    # function to collate several samples of a batch
    new_batch = [] # I will put all samples in this new list (flatten everything) and then call the original collater function
    # TODO replace loops by new_batch = [e for tup in batch for e in tup] TEST FIRST
    if isinstance(batch[0], tuple): # each element of the batch should be a tuple with a main and a sub-question or two independent questions
        for elem in batch:
            for sample in elem:
                new_batch.append(sample)
    return collater(new_batch)

def get_classif_loader(subset, config, shuffle=False):

    # create visual dataset for images
    dataset_visual = visual.get_visual_dataset(subset, config)
    dataset_mask = visual.get_mask_dataset(subset, config, 'maskA')
    dataset_classif = classif.ClassificationDataset(subset, config, dataset_visual, dataset_mask)

    dataloader = DataLoader(    dataset_classif,
                                batch_size = config['batch_size'],
                                shuffle=shuffle,
                                num_workers=config['num_workers'],
                                pin_memory=config['pin_memory'],
                                collate_fn=collater
                            )
    return dataloader




def get_vqa_loader(subset, config, shuffle=False):
    """Function to get a VQA loader. Creates a visual dataset, then the VQA dataset and then the dataloader"""

    # create visual dataset for images
    dataset_visual = visual.get_visual_dataset(subset, config)

    # create vqa dataset depending on dataset name
    if config['dataset']=='vqa2':
        dataset_vqa = vqa.VQA2(subset, config, dataset_visual)
    elif config['dataset']=='vqa2introspect':
        dataset_vqa = vqa.VQA2MainSub(subset, config, dataset_visual)
    elif config['dataset']=='gqa':
        dataset_vqa = vqa.GQAMainSub(subset, config, dataset_visual)
    elif 'idrid_regions_single' in config['dataset'] or 'vqa2_regions_single' in config['dataset']:
        dataset_mask = visual.get_mask_dataset(subset, config, 'maskA')
        if config['mainsub'] and subset != 'test': # if main and sub-questions, use corresponding dataset class
            dataset_vqa = vqa.VQARegionsSingleMainSub(subset, config, dataset_visual, dataset_mask)
        elif config['dataset']=='vqa2_regions_single':
            dataset_vqa = vqa.VQA2RegionsSingle(subset, config, dataset_visual, dataset_mask)
        else:
            dataset_vqa = vqa.VQARegionsSingle(subset, config, dataset_visual, dataset_mask)
    elif 'idrid_regions_complementary' in config['dataset']:
        dataset_maskA = visual.get_mask_dataset(subset, config, 'maskA')
        dataset_maskB = visual.get_mask_dataset(subset, config, 'maskB')
        dataset_vqa = vqa.VQARegionsComplementary(subset, config, dataset_visual, dataset_maskA, dataset_maskB)
    elif 'idrid_regions_dual' in config['dataset']:
        pass
    else:
        raise ValueError("Unknown dataset")

    if config['mainsub'] and subset != 'test':
        collater_fn = collater_mainsub
    else:
        collater_fn = collater

    dataloader = DataLoader(    dataset_vqa,
                                batch_size = config['batch_size'],
                                shuffle=shuffle,
                                num_workers=config['num_workers'],
                                pin_memory=config['pin_memory'],
                                collate_fn=collater_fn
                            )
    if subset == 'train':
        return dataloader, dataset_vqa.map_index_word, dataset_vqa.map_index_answer, dataset_vqa.index_unknown_answer
    else:
        return dataloader

def get_visual_loader(subset, config, shuffle=False):

    # create visual dataset for images
    dataset_visual = visual.get_visual_dataset(subset, config)

    dataloader = DataLoader(dataset_visual, 
                            batch_size= config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'], 
                            pin_memory=config['pin_memory']
                            )
    return dataloader

