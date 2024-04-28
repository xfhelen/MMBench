import torch
import sys
import os 
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import random
import argparse
import yaml
import torch.nn as nn
import torchvision
from transformers import BertModel,BertTokenizer
import librosa
from models.unimodals.common_models import LeNet, MLP
from models.unimodals.robotics.encoders import (ProprioEncoder, ForceEncoder, ImageEncoder, DepthEncoder, ActionEncoder)
from models.fusions.common_fusions import Concat
from models.eval_scripts.complexity import all_in_one_train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', default="normal", type=str, help='mode')
    parser.add_argument('--path_config', default='applications/customization/config.yaml', type=str, help='path to a yaml options file')
    args = parser.parse_args()
    return args

args = args_parser()
options = args.options

class MMDL(nn.Module):
    """Implements MMDL classifier."""
    
    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        if options == "normal":
            outs = []
            if self.has_padding:
                for i in range(len(inputs[0])):
                    outs.append(self.encoders[i]([inputs[0][i], inputs[1][i]]))
                    # print("shape", outs[i].shape)
            else:
                for i in range(len(inputs)):
                    outs.append(self.encoders[i](inputs[i]))
                    print("i={}, shape={}".format(i, outs[i].shape))


            # self.reps = outs
            # if self.has_padding:
            #     if isinstance(outs[0], torch.Tensor):
            #         out = self.fuse(outs)
            #     else:
            #         out = self.fuse([i[0] for i in outs])
            # else:
            #     out = self.fuse(outs)

            # self.fuseout = out
            # if type(out) is tuple:
            #     out = out[0]
    
            # # print("out", out.shape)
            # if self.has_padding and not isinstance(outs[0], torch.Tensor):
            #     return self.head([out, inputs[1][0]])
            # return self.head(out)
            
        # TODO 后面这些还得改
        # elif options == "encoder" :
        #     outs = []
        #     if self.has_padding:
        #         for i in range(len(inputs[0])):
        #             outs.append(self.encoders[i]([inputs[0][i], inputs[1][i]]))
        #     else:
        #         for i in range(len(inputs)):
        #             outs.append(self.encoders[i](inputs[i]))
        #     return outs

        # elif options == "fusion" :
        #     outs = []
        #     outs.append(torch.zeros([40,48]).to(device))
        #     outs.append(torch.zeros([40,192]).to(device))
        #     if self.has_padding:
                
        #         if isinstance(outs[0], torch.Tensor):
        #             out = self.fuse(outs)
        #         else:
        #             out = self.fuse([i[0] for i in outs])
        #     else:
        #         out = self.fuse(outs)
        #         return out
        
        # elif options == "head" :
        #     outs = []
        #     outs.append(torch.zeros([40,48]).to(device))
        #     outs.append(torch.zeros([40,192]).to(device))
        #     out = torch.zeros([40,96].to(device))
        #     if self.has_padding and not isinstance(outs[0], torch.Tensor):
        #         return self.head([out, inputs[1][0]])
        #     return self.head(out)

class Bert_encoder(nn.Module):
    def __init__(self):
        super(Bert_encoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded_input = tokenizer(inputs.to(device), return_tensors='pt')
        output = self.bert(**encoded_input)['last_hidden_state']
        return output[:,0, :]

class Librosa_encoder(nn.Module):
    def __init__(self):
        super(Librosa_encoder, self).__init__()
    
    def forward(self, inputs_tensor):
        inputs_np = inputs_tensor.to('cpu').numpy()
        inputs = (np.array(inputs_np[:-1]), inputs_np[-1])

        y, sr = inputs
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
        jump = int(audio_feature.shape[1] / 10)
        temp_feature = librosa.util.sync(audio_feature, range(1, audio_feature.shape[1], jump))
        audio_feature = torch.tensor(temp_feature).to(device)
        return torch.mean(audio_feature, dim=1).view(1, -1)
        

transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])    

def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.eval()
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152

def train(encoders, fusion, head, valid_dataloader, total_epochs, is_packed=False,input_to_float=True, track_complexity=True):

    model = MMDL(encoders, fusion, head, has_padding=is_packed).to(device)

    def _trainprocess():
        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for _ in range(total_epochs):
            model.eval()
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/customization'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    for j in valid_dataloader:
                        if is_packed:
                            _ = model([[_processinput(i).to(device) for i in j[0]], j[1]])
                        else:
                            _ = model([_processinput(i).to(device) for i in j[:-1]])
                        prof.step()

    if track_complexity:
        all_in_one_train(_trainprocess, [model])
    else:
        _trainprocess()

def main():
    with open(args.path_config, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    encoders = []
    inputs = []
    if not config["have_time"]:
        if config['have_img'] :
            img = torch.zeros([1, config["channels"], config["img_size_x"], config["img_size_y"]]).to(device)
            if config["img_encoder"] == "Lenet":
                Lenet_channels = 6
                encoders.append(LeNet(config["channels"], Lenet_channels , 3).to(device))
                inputs.append(img)
            elif config["img_encoder"] == "Resnet":
                img = transforms(img)
                resnet = pretrained_resnet152().to(device)
                class Identity(torch.nn.Module):
                    def forward(self, input_: torch.Tensor) -> torch.Tensor:
                        return input_
                resnet.fc = Identity()  # Trick to avoid computing the fc1000 layer, as we don't need it here.
                encoders.append(resnet)
                inputs.append(img)
        if config['have_text']:
            article = open(config["text_init_file"]).read()
            sentences = article.split("\n") 
            text = sentences[0]
            if config["text_encoder"] == "Bert":
                encoders.append(Bert_encoder().to(device))
                inputs.append(text)
        if config['have_audio']:
            audio = librosa.load(config["audio_init_file"])
            audio = torch.concat((torch.tensor(audio[0]), torch.tensor([audio[1]])), dim=0)
            if config["audio_encoder"] == "Librosa":
                encoders.append(Librosa_encoder().to(device))
                inputs.append(audio)
        if config['have_sensor']:
            if config["have_force"]:
                force_data = torch.zeros([64, 6, 32]).to(device)
                encoders.append(ForceEncoder(config['zdim'], alpha=config['force']).to(device))
                inputs.append(force_data)
            if config["have_proprio"]:
                proprio_data = torch.zeros([64, 8]).to(device)
                encoders.append(ProprioEncoder(config['zdim'], alpha=config['proprio']).to(device))
                inputs.append(proprio_data)
            if config["have_depth"]:
                depth_data = torch.zeros([64, 1, 128, 128]).to(device)
                encoders.append(DepthEncoder(config['zdim'], alpha=config['depth']).to(device))
                inputs.append(depth_data)
            if config["have_action"]:
                action_data = torch.zeros([64, 4]).to(device)
                encoders.append(ActionEncoder(config['action_dim']).to(device))
                inputs.append(action_data)
        inputs.append(torch.zeros(1)) # label
        inputs = [inputs]

    if config["fusion_type"] == "concat":
        fusion = Concat()

    if config["head_type"] == "MLP":
        head = MLP(config["input_dim"], config["hidden_dim"], config["output_dim"]).to(device)
    
    train(encoders, fusion, head, inputs, total_epochs=1)
                


if __name__ == "__main__":
    main()
