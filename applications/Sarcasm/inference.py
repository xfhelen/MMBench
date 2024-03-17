import argparse
from config import CONFIG_BY_KEY, Config
import json
import os 
import subprocess
import PIL.Image
import torch
import torch.utils.data
from typing import Callable, Dict,Sequence
from torch import nn
import h5py
# from overrides import overrides
import torch
import torch.nn
import torch.utils.data
import torchvision
from torch.nn import functional as F
from tqdm import tqdm
import sys
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import librosa
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from transformers import BertModel,BertTokenizer
from sklearn.preprocessing import StandardScaler
import warnings
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-key", default="", choices=list(CONFIG_BY_KEY))
    parser.add_argument('--options', default="normal", type=str, help='mode')
    return parser.parse_args()

def get_librosa_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path)

    hop_length = 512  # Set the hop length; at 22050 Hz, 512 samples ~= 23ms

    # Remove vocals first
    D = librosa.stft(y, hop_length=hop_length)
    S_full, phase = librosa.magphase(D)

    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine")

    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 2, 4
    power = 2
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    S_foreground = mask_v * S_full

    # Recreate vocal_removal y
    new_D = S_foreground * phase
    y = librosa.istft(new_D)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Compute MFCC features from the raw signal
    mfcc_delta = librosa.feature.delta(mfcc)  # And the first-order differences (delta features)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_delta = librosa.feature.delta(S)

    spectral_centroid = librosa.feature.spectral_centroid(S=S_full)

    audio_feature = np.vstack((mfcc, mfcc_delta, S, S_delta, spectral_centroid))  # combine features

    # binning data
    jump = int(audio_feature.shape[1] / 10)
    return librosa.util.sync(audio_feature, range(1, audio_feature.shape[1], jump))


def save_audio_features(filename) -> None:
    id_ = filename.rsplit(".", maxsplit=1)[0]
    return get_librosa_features(filename)

def to_one_hot(data: Sequence[int], size: int) -> np.ndarray:
    """
    Returns one hot label version of data
    """
    one_hot_data = np.zeros((len(data), size))
    one_hot_data[range(len(data)), data] = 1

    assert np.array_equal(data, np.argmax(one_hot_data, axis=1))
    return one_hot_data

def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.eval()
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152

class Sarcasm(nn.Module):
    def __init__(self):
        super(Sarcasm,self).__init__()
        self.fc1 = nn.Linear(3099, 10)
        self.fc2 = nn.Linear(10, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        output  = self.fc2(x)
        return output

class SarcasmDataset(Dataset):
  def __init__(self, data, labels):
     self.data = data
     self.labels = labels

  def __getitem__(self, index):
     return self.data[index], self.labels[index]

  def __len__(self):
     return len(self.data)

def main() -> None:
    args = parse_args()
    print("Args:", args)
    config = CONFIG_BY_KEY[args.config_key]
    options = args.options
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if options == 'encoder' or options == 'normal': 

        utterence_videos_folder_path = "applications/Sarcasm/data/videos/utterances_final"
        context_videos_folder_path = "applications/Sarcasm/data/videos/context_final"
        utterance_frame_folder_path = "applications/Sarcasm/data/frames/utterences"
        context_frame_folder_path = "applications/Sarcasm/data/frames/context"
        audios_folder_path = "applications/Sarcasm/data/audios/utterances"
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])    


        text = open("applications/Sarcasm/data/bert-input.txt").read()
        sentences = text.split("\n") 

        with open("applications/Sarcasm/data/sarcasm_data.json") as file:
            videos_data_dict = json.load(file)
        
        BERT_PATH = 'applications/Sarcasm/bert-base-uncased'

        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

        bert = BertModel.from_pretrained(BERT_PATH)

        num = 0
        train_input = None
        train_output = []
        total_len = len(os.listdir(utterence_videos_folder_path))
        for video_id in list(videos_data_dict):

            utterance_video_file_path = utterence_videos_folder_path + "/" + video_id + ".mp4"
            context_video_file_path = context_videos_folder_path + "/" + video_id + "_c.mp4"
            utterance_frame_file_path = utterance_frame_folder_path + "/" + video_id
            context_frame_file_path = context_frame_folder_path + "/" + video_id + "_c"
            audio_file_path = audios_folder_path + "/" + video_id + ".aac"
            if not os.path.exists(audios_folder_path):
                os.makedirs(audios_folder_path)
                
            if not os.path.exists(utterance_frame_file_path):
                os.makedirs(utterance_frame_file_path)
                subprocess.run(["ffmpeg", "-i", utterance_video_file_path,  f"{utterance_frame_file_path}/%05d.jpg"])

            if not os.path.exists(context_frame_file_path):
                os.makedirs(context_frame_file_path)
                subprocess.run(["ffmpeg", "-i", context_video_file_path,  f"{context_frame_file_path}/%05d.jpg"])

            if not os.path.exists(audio_file_path):
                subprocess.run(["ffmpeg", "-i", utterance_video_file_path, "-vn", "-acodec", "copy", audio_file_path])

            frames = None

            ## video feature

            for i, frame_file_name in enumerate(os.listdir(utterance_frame_file_path)):
                frame = PIL.Image.open(os.path.join(utterance_frame_file_path, frame_file_name))
                frame = transforms(frame)
                if frames is None:
                    frames = torch.empty((len(os.listdir(utterance_frame_file_path)), *frame.size()))  # noqa
                frames[i] = frame  # noqa

            batch_size = 32
            resnet = pretrained_resnet152().to(DEVICE)
            class Identity(torch.nn.Module):
                def forward(self, input_: torch.Tensor) -> torch.Tensor:
                    return input_

            resnet.fc = Identity()  # Trick to avoid computing the fc1000 layer, as we don't need it here.
            pool5_features_file = torch.empty((0,2048),device=DEVICE)
            for start_index in range(0, len(frames), batch_size):
                end_index = min(start_index + batch_size, len(frames))
                frame_ids_range = range(start_index, end_index)
                frame_batch = frames[frame_ids_range].to(DEVICE)
                avg_pool_value = resnet(frame_batch)
                pool5_features_file  = torch.cat((pool5_features_file, avg_pool_value), axis=0)
            video_feature = torch.mean(pool5_features_file,dim=0).view(-1,2048)

            ## audio feature

            audio_feature = torch.tensor(save_audio_features(audio_file_path)).to(DEVICE)
            audio_feature = torch.mean(audio_feature,dim=1).view(1,-1)

            ## text feature

            encoded_input = tokenizer(sentences[num], return_tensors='pt')
            output = bert(**encoded_input)['last_hidden_state']
            text_feature = output[:,0, :].to(DEVICE)
            
            if options == 'encoder':
                break
            print(video_feature.shape)
            print(audio_feature.shape)
            print(text_feature.shape)
            fused_feature = torch.cat((video_feature,audio_feature,text_feature),axis = 1)

            if train_input is None:
                train_input = fused_feature
            else :
                train_input = torch.cat((train_input,fused_feature),axis = 0)

            train_output.append(videos_data_dict[video_id]['sarcasm'])
            num += 1

            break

        if options == 'normal':
            train_input = train_input.to(torch.float32).detach()
            train_output  = torch.tensor(train_output, dtype=torch.long).to(DEVICE).detach()

    ## train
    # train_input, val_input, train_output, val_output = train_test_split(train_input, train_output, test_size=0.2, random_state=42)
    # train_dataset = SarcasmDataset(train_input,train_output)
    # val_dataset = SarcasmDataset(val_input,val_output)
    # train_dataloader = DataLoader(train_dataset,32)
    # val_dataloader = DataLoader(val_dataset,1)

    # model = Sarcasm().to(DEVICE)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # num_epochs = 1

    # for epoch in range(num_epochs):
    #     model.train()
    #     for idx,(x,y) in enumerate(train_dataloader,0):
    #         # 前向传播
    #         output = model(x)

    #         loss = criterion(output, y)

    #         # 反向传播和优化
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if idx % 10 ==0:
    #             print("epoch={}/{},{}/{} of train, loss={}".format(epoch, 5, idx, len(train_dataloader),loss.item()))
            
    #     model.eval()
    #     total_loss = 0
    #     for idx,(data_x,data_y) in enumerate(val_dataloader,0):
    #         data_x = data_x
    #         data_y = data_y
    #         outputs = model(data_x)
    #         loss = criterion(outputs, data_y)
    #         total_loss += loss.item()
    #     print("val_loss={}".format(total_loss / total_len))
    # torch.save(model.state_dict(), 'model.pt')

    # test
            test_dataset = SarcasmDataset(train_input,train_output)
            test_dataloader = DataLoader(test_dataset,1)
            model = Sarcasm().to(DEVICE)
            model.eval()
            for idx,(data_x,data_y) in enumerate(test_dataloader,0):
                data_x = data_x
                data_y = data_y
                outputs = model(data_x)
    if options == 'fusion':
        video_feature = torch.ones(1, 2048).to(DEVICE)
        audio_feature = torch.ones(1, 283).to(DEVICE)
        text_feature = torch.ones(1, 768).to(DEVICE)
        fused_feature = torch.cat((video_feature,audio_feature,text_feature),axis = 1)
    if options == 'head':
        train_input = torch.ones(1,3099).to(DEVICE)
        model = Sarcasm().to(DEVICE)
        model.eval()
        outputs = model(train_input)
    


if __name__ == "__main__":
    main()
