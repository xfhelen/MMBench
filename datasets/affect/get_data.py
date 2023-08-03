"""Implements dataloaders for AFFECT data."""
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import sys
from typing import *
import numpy as np
from torch.nn import functional as F
from datasets.affect.data_preprocessing import data_preprocess_fast

sys.path.append(os.getcwd())


np.seterr(divide='ignore', invalid='ignore')


def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    # for ind, k in enumerate(dataset["vision"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    # for ind, k in enumerate(dataset["audio"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)

    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def z_norm(dataset, max_seq_len=50):
    """Normalize data in the dataset."""
    processed = {}
    text = dataset['text'][:, :max_seq_len, :]
    vision = dataset['vision'][:, :max_seq_len, :]
    audio = dataset['audio'][:, :max_seq_len, :]
    for ind in range(dataset["text"].shape[0]):
        vision[ind] = np.nan_to_num(
            (vision[ind] - vision[ind].mean(0, keepdims=True)) / (np.std(vision[ind], axis=0, keepdims=True)))
        audio[ind] = np.nan_to_num(
            (audio[ind] - audio[ind].mean(0, keepdims=True)) / (np.std(audio[ind], axis=0, keepdims=True)))
        text[ind] = np.nan_to_num(
            (text[ind] - text[ind].mean(0, keepdims=True)) / (np.std(text[ind], axis=0, keepdims=True)))

    processed['vision'] = vision
    processed['audio'] = audio
    processed['text'] = text
    processed['labels'] = dataset['labels']
    return processed


class Affectdataset(Dataset):
    """Implements Affect data as a torch dataset."""

    def __init__(self, data: Dict, flatten_time_series: bool, aligned: bool = True, task: str = None, max_pad=False, max_pad_num=50, data_type='mosi', z_norm=False) -> None:
        """Instantiate AffectDataset

        Args:
            data (Dict): Data dictionary
            flatten_time_series (bool): Whether to flatten time series or not
            aligned (bool, optional): Whether to align data or not across modalities. Defaults to True.
            task (str, optional): What task to load. Defaults to None.
            max_pad (bool, optional): Whether to pad data to max_pad_num or not. Defaults to False.
            max_pad_num (int, optional): Maximum padding number. Defaults to 50.
            data_type (str, optional): What data to load. Defaults to 'mosi'.
            z_norm (bool, optional): Whether to normalize data along the z-axis. Defaults to False.
        """
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task
        self.max_pad = max_pad
        self.max_pad_num = max_pad_num
        self.data_type = data_type
        self.z_norm = z_norm
        self.dataset['audio'][self.dataset['audio'] == -np.inf] = 0.0

    def __getitem__(self, ind):
        """Get item from dataset."""
        # vision = torch.tensor(vision)
        # audio = torch.tensor(audio)
        # text = torch.tensor(text)

        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = torch.tensor(self.dataset['text'][ind])

        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
                # start = 0
            except:
                print(text, ind)
                exit()
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()

        # z-normalize data
        if self.z_norm:
            vision = torch.nan_to_num((vision - vision.mean(0, keepdims=True)) /
                                      (torch.std(vision, axis=0, keepdims=True)))
            audio = torch.nan_to_num((audio - audio.mean(0, keepdims=True)) / (torch.std(audio, axis=0, keepdims=True)))
            text = torch.nan_to_num((text - text.mean(0, keepdims=True)) / (torch.std(text, axis=0, keepdims=True)))

        def _get_class(flag, data_type=self.data_type):
            # print("DEBUG:",flag)
            if data_type in ['mosi', 'mosei', 'sarcasm']:
                if flag > 0:
                    return [[1]]
                else:
                    return [[0]]
            else:
                return [flag]

        tmp_label = self.dataset['labels'][ind]
        if self.data_type == 'humor' or self.data_type == 'sarcasm':
            if (self.task == None) or (self.task == 'regression'):
                if self.dataset['labels'][ind] < 1:
                    tmp_label = [[-1]]
                else:
                    tmp_label = [[1]]
        else:
            tmp_label = self.dataset['labels'][ind]

        label = torch.tensor(_get_class(tmp_label)).long(
        ) if self.task == "classification" else torch.tensor(tmp_label).float()

        if self.flatten:
            return [vision.flatten(), audio.flatten(), text.flatten(), ind,
                    label]
        else:
            if self.max_pad:
                tmp = [vision, audio, text, label]
                for i in range(len(tmp) - 1):
                    tmp[i] = tmp[i][:self.max_pad_num]
                    tmp[i] = F.pad(tmp[i], (0, 0, 0, self.max_pad_num - tmp[i].shape[0]))
            else:
                tmp = [vision, audio, text, ind, label]
            return tmp

    def __len__(self):
        """Get length of dataset."""
        return self.dataset['vision'].shape[0]


def get_dataloader(
        rawdata_dir: str, batch_size: int = 32, max_seq_len=50, max_pad=False, num_workers: int = 2, flatten_time_series: bool = False, task=None, z_norm=False) -> DataLoader:
    """Get dataloaders for affect data.

    Args:
        filepath (str): Path to datafile
        batch_size (int, optional): Batch size. Defaults to 32.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 50.
        max_pad (bool, optional): Whether to pad data to max length or not. Defaults to False.
        num_workers (int, optional): Number of workers. Defaults to 2.
        flatten_time_series (bool, optional): Whether to flatten time series data or not. Defaults to False.
        task (str, optional): Which task to load in. Defaults to None.
        z_norm (bool, optional): Whether to normalize data along the z dimension or not. Defaults to False.

    Returns: validation dataloader
    """
    # with open(filepath, "rb") as f:
    #     alldata = pickle.load(f)
    alldata = data_preprocess_fast(label_path= rawdata_dir + 'CMU_MOSEI_Labels.csd', audio_path=rawdata_dir + 'CMU_MOSEI_COVAREP.csd', vision_path=rawdata_dir + 'CMU_MOSEI_VisualOpenFace2.csd', text_path=rawdata_dir + 'CMU_MOSEI_TimestampedWordVectors.csd', output_file_path = 'datasets/affect/mosei_new.pkl')

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
    alldata['valid'] = drop_entry(alldata['valid'])
    
    process = eval("_process_2") if max_pad else eval("_process_1")
    
    processed_dataset['valid'] = alldata['valid']
    
    valid = DataLoader(
        Affectdataset(processed_dataset['valid'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type='mosei', z_norm=z_norm),
        shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=process)

    return valid


def _process_1(inputs: List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0]) - 2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(pad_seq)

    for sample in inputs:

        inds.append(sample[-2])
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input, processed_input_lengths, \
        torch.tensor(inds).view(len(inputs), 1), torch.tensor(labels).view(len(inputs), 1)


def _process_2(inputs: List):
    processed_input = []
    processed_input_lengths = []
    labels = []

    for i in range(len(inputs[0]) - 1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        processed_input.append(torch.stack(feature))

    for sample in inputs:
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input[0], processed_input[1], processed_input[2], torch.tensor(labels).view(len(inputs), 1)
