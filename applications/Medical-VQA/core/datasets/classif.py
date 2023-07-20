# Project:
#   VQA
# Description:
#   Dataset class for a classifier
# Author: 
#   Sergio Tascon-Morales

from os.path import join as jp
import pickle
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, subset, config, dataset_visual, dataset_mask):
        self.subset = subset 
        self.config = config
        self.dataset_visual = dataset_visual
        self.dataset_mask = dataset_mask

        self.path_classif = jp(config['path_qa'], 'classif')

        self.read_data(self.path_classif)

    def read_data(self, path_files):
        path_dataset = jp(path_files, self.subset + 'set.pickle')
        with open(path_dataset, 'rb') as f:
                    self.dataset_classif = pickle.load(f)

    def __getitem__(self, index):
        sample = {}

        # get qa pair
        item = self.dataset_classif[index]

        # get visual
        visual = self.dataset_visual.get_by_name(item['image'])['visual']
        mask = self.dataset_mask.get_by_name(item['mask'])['mask']

        sample['visual'] = visual
        sample['maskA'] = mask

        if self.subset == 'test':
            sample['label'] = item['answer']
        else:
            sample['label'] = item['answer']
            sample['index_gt'] = item['index_gt']

        return sample

    def __len__(self):
        return len(self.dataset_classif)