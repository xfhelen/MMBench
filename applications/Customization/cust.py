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
from transformers import BertModel,BertTokenizer,BertConfig,RobertaModel, RobertaTokenizer, RobertaConfig,DistilBertModel, DistilBertTokenizer, DistilBertConfig,GPT2Model, GPT2Tokenizer, GPT2Config
import librosa
from models.unimodals.common_models import LeNet, MLP,TransformerWithMlp
from models.unimodals.robotics.encoders import (ProprioEncoder, ForceEncoder, ImageEncoder, DepthEncoder, ActionEncoder)
from models.fusions.common_fusions import Concat,TensorFusion,AttentionFusion,RNNFusion,LowRankTensorFusion,create_film_layer,create_film_layer_v2,ConcatWithLinear,MultiplicativeInteractions2Modal,MultiplicativeInteractions3Modal
from torchvggish import vggish, vggish_input  # 添加 vggish_input 的导入
from torchaudio.models import Wav2Vec2Model
import warnings
import opensmile
import cv2
import torchvision.transforms as transforms

# 抑制所有警告
warnings.filterwarnings("ignore")

# from models.eval_scripts.complexity import all_in_one_train
# from models.utils.helper_modules import Sequential2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', default="normal", type=str, help='mode')
    parser.add_argument('--path_config', default='applications/Customization/config.yaml', type=str, help='path to a yaml options file')
    parser.add_argument('--torchprofiler', action='store_true', help='Enable profiler mode')
    parser.add_argument('--intelv', action='store_true', help='Enable intel vtune mode')
    args = parser.parse_args()
    return args

args = args_parser()
options = args.options
torchprofiler = args.torchprofiler
intel_vtune = args.intelv
if intel_vtune:
    torch.cuda.is_available = lambda : False
    print("Using CPU only mode")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MMDL(nn.Module):
    
    def __init__(self, encoders, fusion, head,fusion_input_dim,head_input_dim):

        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.fuseout = None
        self.reps = []
        self.fusion_input_dim = fusion_input_dim
        self.head_input_dim = head_input_dim

    def forward(self, inputs):
        if options == "normal":
            outs = []
            print("encoder stage")
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
                print("i={}, shape={}".format(i + 1 , outs[i].shape))

            self.reps = outs
            out = self.fuse(outs)

            print("fusion stage")
            print("out", out.shape)

            self.fuseout = out

            # if self.has_padding and not isinstance(outs[0], torch.Tensor):
            #     return self.head([out, inputs[1][0]])
            out = self.head(out.float())
            print("head stage")
            print("out", out.shape)
            return out
            

        elif options == "encoder" :
            outs = []
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
            return outs
                    

        elif options == "fusion" :
            outs = []
            for i in self.fusion_input_dim:
                outs.append(torch.zeros([1, i]).to(device))

            out = self.fuse(outs)
            return out
        
        elif options == "head" :
            out = torch.zeros(self.head_input_dim).to(device)
            # if self.has_padding and not isinstance(outs[0], torch.Tensor):
            #     return self.head([out, inputs[1][0]])
            return self.head(out)

class RandomBertTokenizer:
    def __init__(self, vocab_size=30522):
        # 创建一个随机词汇表
        self.vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.vocab_size = vocab_size

    def __call__(self, texts, return_tensors='pt', padding=True, truncation=True):
        # 简单的分词逻辑，将每个字符映射到一个随机的 token
        encoded_inputs = []
        for text in texts:
            encoded_input = [self.vocab.get(f"token_{ord(char) % self.vocab_size}", 0) for char in text]
            encoded_inputs.append(encoded_input)
        
        # 将输入转换为张量，并进行填充和截断
        max_length = max(len(seq) for seq in encoded_inputs)
        if padding:
            encoded_inputs = [seq + [0] * (max_length - len(seq)) for seq in encoded_inputs]
        if truncation:
            encoded_inputs = [seq[:max_length] for seq in encoded_inputs]
        
        return torch.tensor(encoded_inputs)

class LSTM_encoder(nn.Module):
    def __init__(self, vocab_size=30522, embedding_dim=768, hidden_dim=512, num_layers=2):
        super(LSTM_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, inputs):
        tokenizer = RandomBertTokenizer()
        encoded_input = tokenizer(inputs).to(device)
        embedded = self.embedding(encoded_input)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = torch.mean(lstm_out, dim=0, keepdim=True)
        return lstm_out[:, -1, :]

class Bert_encoder(nn.Module):
    def __init__(self):
        super(Bert_encoder, self).__init__()
        config = BertConfig()
        self.bert = BertModel(config).to(device)
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        tokenizer = RandomBertTokenizer()
        encoded_input = tokenizer(inputs).to(device)
        output = self.bert(encoded_input)["last_hidden_state"]
        return output[0,:, :]

class Roberta_encoder(nn.Module):
    def __init__(self):
        super(Roberta_encoder, self).__init__()
        config = RobertaConfig()
        self.roberta = RobertaModel(config).to(device)
        self.roberta.eval()
        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        tokenizer = RobertaTokenizer.from_pretrained('applications/Customization/roberta-base')
        encoded_input = tokenizer(inputs, return_tensors='pt').to(device)
        output = self.roberta(**encoded_input)['last_hidden_state']
        return output[:, 0, :]
    
class DistilBert_encoder(nn.Module):
    def __init__(self):
        super(DistilBert_encoder, self).__init__()
        config = DistilBertConfig()
        self.distilbert = DistilBertModel(config).to(device)
        self.distilbert.eval()
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        tokenizer = DistilBertTokenizer.from_pretrained('applications/Customization/distilbert-base-uncased')
        encoded_input = tokenizer(inputs, return_tensors='pt').to(device)
        output = self.distilbert(**encoded_input)['last_hidden_state']
        return output[:, 0, :]
    
class GPT2_encoder(nn.Module):
    def __init__(self):
        super(GPT2_encoder, self).__init__()
        config = GPT2Config()
        self.gpt2 = GPT2Model(config).to(device)
        self.gpt2.eval()
        for param in self.gpt2.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        tokenizer = GPT2Tokenizer.from_pretrained('applications/Customization/gpt2')
        encoded_input = tokenizer(inputs, return_tensors='pt').to(device)
        output = self.gpt2(**encoded_input)['last_hidden_state']
        return output[:, 0, :]

class Librosa_encoder(nn.Module):
    def __init__(self):
        super(Librosa_encoder, self).__init__()
    
    def forward(self, inputs_file):
        audio = librosa.load(inputs_file)
        inputs_tensor = torch.concat((torch.tensor(audio[0]), torch.tensor([audio[1]])), dim=0)
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

class VGGish_encoder(nn.Module):
    def __init__(self):
        super(VGGish_encoder, self).__init__()
        self.vggish = vggish().to(device)  # 确保模型在正确的设备上

    def forward(self, inputs_file):
        wav_preprocess = vggish_input.wavfile_to_examples(inputs_file)  # 保持默认设备
        wav_preprocess = wav_preprocess.unsqueeze(dim=1)
        input_wav = wav_preprocess.float()  # 保证输入为 float 类型

        # 确保输入数据和模型在同一设备
        input_wav = input_wav.to(self.vggish.parameters().__next__().device)  # 直接将输入数据转移到模型所在的设备

        with torch.no_grad():
            output = self.vggish(input_wav.squeeze(2))  # 输入数据已经在正确的设备
            output = output.squeeze(0)

        return output


class OpenSMILE_encoder(nn.Module):
    def __init__(self):
        super(OpenSMILE_encoder, self).__init__()
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def forward(self, inputs_file):
        audio, sr = librosa.load(inputs_file, sr=None)
        inputs_tensor = torch.concat((torch.tensor(audio), torch.tensor([sr])), dim=0).to(device)
        inputs_np = inputs_tensor.to('cpu').numpy()
        inputs = (np.array(inputs_np[:-1]), inputs_np[-1])

        y, sr = inputs
        features = self.smile.process_signal(y, sr)
        features = torch.tensor(features.values).to(device)
        return features.view(1, -1)

class Wav2Vec2_encoder(nn.Module):
    def __init__(self):
        super(Wav2Vec2_encoder, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("applications/Customization/wav2vec2-base-960h").to(device)
        self.wav2vec2.eval()
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward(self, inputs_file):
        audio, sr = librosa.load(inputs_file, sr=None)
        inputs_tensor = torch.concat((torch.tensor(audio), torch.tensor([sr])), dim=0).to(device)
        inputs_np = inputs_tensor.to('cpu').numpy()
        inputs = (np.array(inputs_np[:-1]), inputs_np[-1])

        y, sr = inputs
        inputs_tensor = torch.tensor(y).unsqueeze(0).to(device)
        with torch.no_grad():
            features = self.wav2vec2(inputs_tensor).last_hidden_state
        return torch.mean(features, dim=1).view(1, -1)

class Linear_out_Lenet(nn.Module):
    def __init__(self, in_channels, args_channels, additional_layers):
        super(Linear_out_Lenet, self).__init__()
        self.lenet = LeNet(in_channels, args_channels, additional_layers)

    def forward(self, x):
        out = self.lenet(x)
        out = torch.flatten(out, start_dim=1)
        out = torch.mean(out, dim=0, keepdim=True)
        return out

class ResNetEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ResNetEncoder, self).__init__()
        # 定义图像预处理步骤
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 加载预训练的 ResNet152 模型
        self.resnet = torchvision.models.resnet152(pretrained=True)
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 替换全连接层
        self.resnet.fc = Identity()
        
        # 添加一个线性层将输出特征维度从 2048 降到 768
        self.linear = nn.Linear(2048, output_dim).to(device)

    def forward(self, img):
        # 检查输入图像的维度
        if (img.dim() == 4):
            # 如果输入是批次图像，取第一个图像
            img = img[0]
        # 将张量转换为 PIL 图像
        img = transforms.ToPILImage()(img)
        # 应用图像预处理
        img = self.transforms(img)
        # 添加批次维度
        img = img.unsqueeze(0).to(device)  # 确保输入数据在同一设备上
        # 通过 ResNet 模型
        features = self.resnet(img)
        features = torch.flatten(features, start_dim=1)
        features = torch.mean(features, dim=0, keepdim=True)
        features = self.linear(features)  # 通过线性层
        return features

class ResNet3DEncoder(nn.Module):
    def __init__(self,num_frames = 16):
        super(ResNet3DEncoder, self).__init__()
        # 定义视频预处理步骤
        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 加载预训练的 ResNet3D 模型
        self.resnet3d = torchvision.models.video.r3d_18(pretrained=True)
        self.resnet3d.eval()
        for param in self.resnet3d.parameters():
            param.requires_grad = False
        
        # 替换全连接层
        self.resnet3d.fc = Identity()
        self.num_frames = num_frames

    def forward(self, video_path):
        # 应用视频预处理
        video = self.load_video_frames(video_path, num_frames=self.num_frames).to(device)
        batch_size, channels, frames, height, width = video.shape
        video = video.view(-1, channels, height, width)  # 展平时间维度
        video = self.transforms(video)
        video = video.view(batch_size, channels, frames, video.shape[-2], video.shape[-1])  # 恢复时间维度
        # 通过 ResNet3D 模型
        features = self.resnet3d(video)
        return features
    
    def load_video_frames(self,video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (171, 128))  # Resize to match the expected input size
            frames.append(frame)
        cap.release()
        
        # 如果视频帧数不足，重复最后一帧
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        frames = np.array(frames)
        frames = frames.transpose(3, 0, 1, 2)  # 转换为 (channels, frames, height, width)
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0  # 归一化到 [0, 1]
        return frames

class Identity(nn.Module):
    def forward(self, input_):
        return input_


def run_profiler(encoders, fusion, head, valid_dataloader, fusion_input_dim,head_input_dim):

    model = MMDL(encoders, fusion, head,fusion_input_dim,head_input_dim).to(device)

    # def _processinput(inp):
    #     if input_to_float:
    #         return inp.float()
    #     else:
    #         return inp
    
    model.eval()
    if torchprofiler:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/customization'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for j in valid_dataloader:
                    # _ = model([_processinput(i).to(device) for i in j[:-1]])

                    prof.step()
    elif intel_vtune:
        with torch.no_grad():
            with torch.autograd.profiler.emit_itt():
                for j,data in enumerate(valid_dataloader):
                    with torch.profiler.itt.range(f'iteration_{j}'):
                        _ = model(data)
    else:
        with torch.no_grad():
            for j in valid_dataloader:
                _ = model(j)


def main():
    with open(args.path_config, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    encoders = []
    inputs = []
    encoder_output_dim = []
    if not config["have_time"]:
        if config['have_img'] :
            img = torch.randn([1, config["channels"], config["img_size_x"], config["img_size_y"]]).to(device)
            inputs.append(img)
            if config["img_encoder"] == "Lenet":

                encoders.append(Linear_out_Lenet(config["channels"], config["Lenet_channels"] , config["Lenet_additional_layers"]).to(device))
                encoder_output_dim.append(config["Lenet_output_dim"])
                
            elif config["img_encoder"] == "Resnet":
                encoders.append(ResNetEncoder(config["Resnet_output_dim"]))
                encoder_output_dim.append(config["Resnet_output_dim"])

        if config['have_video'] :
            inputs.append(config['video_init_file'])
            encoders.append(ResNet3DEncoder(config['num_frames']))

        if config['have_text']:
            article = open(config["text_init_file"]).read()
            sentences = article.split("\n") 
            text = sentences[0]
            inputs.append(text)
            if config["text_encoder"] == "Roberta":
                encoders.append(Roberta_encoder().to(device))
                encoder_output_dim.append(config["Bert_output_dim"])

            elif config["text_encoder"] == "DistilBert":
                encoders.append(DistilBert_encoder().to(device))
                encoder_output_dim.append(config["Bert_output_dim"])

            elif config["text_encoder"] == "GPT2":
                encoders.append(GPT2_encoder().to(device))
                encoder_output_dim.append(config["Bert_output_dim"])

            elif config["text_encoder"] == "Bert":
                encoders.append(Bert_encoder().to(device))
                encoder_output_dim.append(config["Bert_output_dim"])
            
            elif config["text_encoder"] == "LSTM":
                encoders.append(LSTM_encoder().to(device))
                encoder_output_dim.append(config["LSTM_output_dim"])

            
            

        if config['have_audio']:
            inputs.append(config["audio_init_file"])

            if config["audio_encoder"] == "Librosa":
                
                encoders.append(Librosa_encoder().to(device))
                encoder_output_dim.append(config["Librosa_output_dim"])

            if config["audio_encoder"] == "VGGish":
                
                encoders.append(VGGish_encoder().to(device))


            if config["audio_encoder"] == "OpenSMILE":
                encoders.append(OpenSMILE_encoder().to(device))
                encoder_output_dim.append(config["OpenSMILE_output_dim"])

            if config["audio_encoder"] == "Wav2Vec2":
                encoders.append(Wav2Vec2_encoder().to(device))
                encoder_output_dim.append(config["Wav2Vec2_output_dim"])
        
        if config['have_sensor']:
            if config["have_force"]:
                force_data = torch.zeros([1, 6, 32]).to(device)
                encoders.append(ForceEncoder(config['zdim'], alpha=config['force']).to(device))
                inputs.append(force_data)
                encoder_output_dim.append(config["force_output_dim"])
            if config["have_proprio"]:
                proprio_data = torch.zeros([1, 8]).to(device)
                encoders.append(ProprioEncoder(config['zdim'], alpha=config['proprio']).to(device))
                inputs.append(proprio_data)
                encoder_output_dim.append(config["proprio_output_dim"])
            if config["have_depth"]:
                depth_data = torch.zeros([1, 1, 128, 128]).to(device)
                encoders.append(DepthEncoder(config['zdim'], alpha=config['depth']).to(device))
                inputs.append(depth_data)
                encoder_output_dim.append(config["depth_output_dim"])
            if config["have_action"]:
                action_data = torch.zeros([1, 4]).to(device)
                encoders.append(ActionEncoder(config['action_dim']).to(device))
                inputs.append(action_data)
                encoder_output_dim.append(config["action_output_dim"])
        
        inputs = [inputs]

    fusion_input_dim = encoder_output_dim
    
    if config["fusion_type"] == "Concat":
        fusion = Concat(config["fusion_concat_dim"])
        fusion_output_dim = [1, sum(fusion_input_dim)]
    if config["fusion_type"] == "Tensor Fusion":
        fusion  = TensorFusion()
        sss=1
        for s in fusion_input_dim:
            sss*=s+1
        fusion_output_dim = [1,sss]
    if config["fusion_type"] == "AttentionFusion":
        fusion_output_dim = [1,config["fusion_output_dim"]]
        fusion  = AttentionFusion(fusion_input_dim,fusion_output_dim[1])
    if config["fusion_type"] == "RNNFusion":
        fusion_output_dim = [1,config["fusion_output_dim"]]
        fusion  = RNNFusion(fusion_output_dim[1])
    if config["fusion_type"] == "LowRankTensorFusion":
        fusion_output_dim = [1,config["fusion_output_dim"]]
        fusion  = LowRankTensorFusion(fusion_input_dim,fusion_output_dim[1],config["fusion_rank"])
    if config["fusion_type"] == "FiLM":
        fusion_output_dim = [1,config["fusion_output_dim"]]
        if len(fusion_input_dim) == 3:
            fusion = create_film_layer_v2(fusion_input_dim[0], fusion_input_dim[1], fusion_input_dim[2], config["fusion_hidden_dim"])
        elif len(fusion_input_dim) == 2:
            fusion = create_film_layer(fusion_input_dim[0], fusion_input_dim[1], config["fusion_hidden_dim"])
        else:
            print("Only support 2or3 modalities in FiLM fusion")
            exit()
    if config["fusion_type"] == "ConcatWithLinear":
        fusion_output_dim = [1, config["fusion_output_dim"]]
        fusion = ConcatWithLinear(sum(fusion_input_dim), config["fusion_output_dim"], config["fusion_concat_dim"])
    if config["fusion_type"] == "MultiplicativeInteractions2Modal":
        fusion_output_dim = [1, config["fusion_output_dim"]]
        fusion = MultiplicativeInteractions2Modal(fusion_input_dim, config["fusion_output_dim"], config["fusion_output_choice"], config["fusion_flatten"], None if config["fusion_clip"]=="None" else config["fusion_clip"], None if config["fusion_grad_clip"]=="None" else config["fusion_grad_clip"], config["fusion_flip"])  
    if config["fusion_type"] == "MultiplicativeInteractions3Modal":
        fusion_output_dim = [1, config["fusion_output_dim"]]
        fusion = MultiplicativeInteractions3Modal(fusion_input_dim, config["fusion_output_dim"], config["fusion_output_choice"], config["fusion_flatten"], None if config["fusion_clip"]=="None" else config["fusion_clip"], None if config["fusion_grad_clip"]=="None" else config["fusion_grad_clip"], config["fusion_flip"])  



    head_input_dim = fusion_output_dim
    if config["head_type"] == "MLP":
        head = MLP(head_input_dim[1], config["head_MLP_hidden_dim"], config["head_output_dim"]).to(device)
        
    if config['head_type'] == "TransformerWithMlp":
        head = TransformerWithMlp(head_input_dim[1], config["head_Transformer_hidden_dim"], config["head_output_dim"], config["head_Transformer_conv"]).to(device)
    run_profiler(encoders, fusion, head, inputs,fusion_input_dim,head_input_dim)
                


if __name__ == "__main__":
    main()
