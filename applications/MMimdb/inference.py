import torch
import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import random
from PIL import Image
import argparse
import torch.nn as nn
softmax = nn.Softmax()
import torchvision.models as models
from models.training_structures.Supervised_Learning import test
from torch.nn import Linear
import json
import sklearn.metrics
import numpy
import re
import sys
from typing import *
import logging
import math
import copy
import pickle
from models.unimodals.common_models import MaxOut_MLP
from models.unimodals.common_models import Linear as mod_linear
from models.fusions.common_fusions import Concat
from transformers import AlbertTokenizer, AlbertModel
import torch
from torch.autograd import Variable
from functools import reduce
import operator


torch.cuda.empty_cache()
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mmimdb_simple_late_fusion', help='name of model')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--seed', type=int, default=20, help='random seed')
    parser.add_argument('--options', type=str, default='normal', help='random seed')
    args = parser.parse_args()
    return args
args= args_parser()
####################################
#
# set configuration according params
#
####################################

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu) 
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)

class VGGClassifier(object):
    def __init__(self,device,model_path='vgg.tar', synset_words='synset_words.txt'):
        pass
    def resize_and_crop_image(input_file, output_box=[224, 224], fit=True):
        # https://github.com/BVLC/caffe/blob/master/tools/extra/resize_and_crop_images.py
        '''Downsample the image.
        '''
        img = Image.open(input_file)
        box = output_box
        # preresize image with factor 2, 4, 8 and fast algorithm
        factor = 1
        while img.size[0] / factor > 2 * box[0] and img.size[1] * 2 / factor > 2 * box[1]:
            factor *= 2
        if factor > 1:
            img.thumbnail(
                (img.size[0] / factor, img.size[1] / factor), Image.NEAREST)

        # calculate the cropping box and get the cropped part
        if fit:
            x1 = y1 = 0
            x2, y2 = img.size
            wRatio = 1.0 * x2 / box[0]
            hRatio = 1.0 * y2 / box[1]
            if hRatio > wRatio:
                y1 = int(y2 / 2 - box[1] * wRatio / 2)
                y2 = int(y2 / 2 + box[1] * wRatio / 2)
            else:
                x1 = int(x2 / 2 - box[0] * hRatio / 2)
                x2 = int(x2 / 2 + box[0] * hRatio / 2)
            img = img.crop((x1, y1, x2, y2))

        # Resize the image with best quality algorithm ANTI-ALIAS
        img = img.resize(box, Image.ANTIALIAS).convert('RGB')
        img = numpy.asarray(img, dtype='float32')[..., [2, 1, 0]]
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1))
        img = numpy.expand_dims(img, axis=0)
        return img


class MMDL_mmimdb(nn.Module):
    """Implements MMDL classifier."""
    
    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMDL_mmimdb, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []
    def forward(self, inputs,options):
        if options == "normal":
            outs = []
            encoded_output=model_transformer(**inputs[0])
            encoded_output=encoded_output.last_hidden_state[:,-1,:]#.float()
            encoded_output=self.encoders[0](encoded_output)
            outs.append(encoded_output)
            acti_layer=vgg16(inputs[1].float())#
            acti_layer=vgg16.add_linear(acti_layer).float()
            acti_layer=self.encoders[1](acti_layer)
            outs.append(acti_layer)

            # print("encoded_output_shape", encoded_output.shape)
            # print("acti_layer", acti_layer.shape)

            self.reps = outs
            if self.has_padding:
                if isinstance(outs[0], torch.Tensor):
                    out = self.fuse(outs)
                else:
                    out = self.fuse([i[0] for i in outs])
            else:
                out = self.fuse(outs)

            self.fuseout = out
            if type(out) is tuple:
                out = out[0]
    
            # print("out", out.shape)
            if self.has_padding and not isinstance(outs[0], torch.Tensor):
                return self.head([out, inputs[1][0]])
            return self.head(out)
            
        elif options == "encoder" :
            outs = []
            encoded_output=model_transformer(**inputs[0])
            encoded_output=encoded_output.last_hidden_state[:,-1,:]#.float()
            encoded_output=self.encoders[0](encoded_output)
            outs.append(encoded_output)
            #acti_layer=self.encoders[1](inputs[1])
            #print("inputs[1]",inputs[1])
            acti_layer=vgg16(inputs[1].float())#
            acti_layer=vgg16.add_linear(acti_layer).float()
            acti_layer=self.encoders[1](acti_layer)
            outs.append(acti_layer)
            return outs

        elif options == "fusion" :
            outs = []
            outs.append(torch.zeros([1,512]))
            outs.append(torch.zeros([1,512]))
            if self.has_padding:
                
                if isinstance(outs[0], torch.Tensor):
                    out = self.fuse(outs)
                else:
                    out = self.fuse([i[0] for i in outs])
            else:
                out = self.fuse(outs)
                return out
        
        elif options == "head" :
            outs = []
            outs.append(torch.zeros([1,512]))
            outs.append(torch.zeros([1,512]))
            out = torch.zeros([1,1024])
            if self.has_padding and not isinstance(outs[0], torch.Tensor):
                return self.head([out, inputs[1][0]])
            return self.head(out)

############################## init parameters ##############################################

criterion=torch.nn.BCEWithLogitsLoss()
task="multilabel"
auprc=False
input_to_float=True
no_robust=True

def _processinput(inp):
    if input_to_float:
        return inp.float()
    else:
        return inp
def normalizeText(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text.split()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


with open("./mapping", "rb") as fp:   # Unpickling
    mapping = pickle.load(fp)
    #print(b)

with open('./list.txt', 'r') as f:
    #files = f.read().splitlines()[18160:25959] ## the test set of mmimdf
    files_origin = f.read().splitlines()#[25859:25959]
files=[]
for i in range(0,25959):
    files.append(files_origin[mapping[i][1]])
# files=files[18160:25959]## the test set
files=files[25950:25959]  ## subset test
labels=np.load("./imdb_res.npy")
logger.info('Reading json and jpeg files...')

vocab_counts = []

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model_transformer = AlbertModel.from_pretrained("albert-base-v2", from_tf=True)

#model_transformer .add_module("MaxOut_MLP0",MaxOut_MLP(512, 512, 768, linear_layer=False))

model_transformer.to(device)
model_transformer.eval()
""""""
clsf = VGGClassifier(device,model_path='vgg16.tar', synset_words='synset_words.txt')

"""
#print("Testing:")
model = torch.load(filename).to(device)
torch.save(model.state_dict(), "best_ef5489_dict.pth")
del model
"""



if args.model_name=="mmimdb_simple_late_fusion":
    encoders = [MaxOut_MLP(512, 512, 768, linear_layer=False),
                MaxOut_MLP(512, 1024, 4096, 512, False)]
    head = Linear(1024, 23).to(device)
    fusion = Concat().to(device)
elif args.model_name=="mmimdb_simple_early_fusion":
    ### mmimdb_simple_early_fusion
    from models.unimodals.common_models import Identity
    encoders = [Identity(), Identity() ]
    head = MaxOut_MLP(23, 512, 4864).to(device)
    fusion = Concat().to(device)
elif args.model_name=="mmimdb_low_rank_tensor":
    ###mmimdb_low_rank_tensor
    from models.fusions.common_fusions import LowRankTensorFusion
    encoders = [MaxOut_MLP(512, 512, 768, linear_layer=False),
                MaxOut_MLP(512, 1024, 4096, 512, False)]
    head = Linear(512, 23).to(device)
    fusion = LowRankTensorFusion([512, 512], 512, 128).to(device)
elif args.model_name=="mmimdb_multi_interac_matrix":
    from models.fusions.common_fusions import MultiplicativeInteractions2Modal
    encoders = [MaxOut_MLP(512, 512, 768, linear_layer=False),
                MaxOut_MLP(512, 1024, 4096, 512, False)]
    head = Linear(1024, 23).cuda()
    fusion = MultiplicativeInteractions2Modal([512, 512], 1024, 'matrix').cuda()
vgg16 = models.vgg16(pretrained=True)
vgg16.add_module("add_linear",Linear(1000,4096))
vgg16.to(device)
vgg16.eval()

model = MMDL_mmimdb(encoders, fusion, head, has_padding=False).to(device)

model.eval()

totalloss = 0.0
pred = []
true = []

options = args.options
############################## load image/text  ##############################################

for i, file in enumerate(files):
    if options == "encoder" or options == "normal":
        with open(file) as f:
            data = json.load(f)
            data['imdb_id'] = file.split('/')[-1].split('.')[0]
            # if 'plot' in data and 'plot outline' in data:
            #    data['plot'].append(data['plot outline'])
            im_file = file.replace('json', 'jpeg')
            if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):
                plot_id = numpy.array([len(p) for p in data['plot']]).argmax()
                
                data['plot'] = data['plot'][plot_id]

                ##################################### tokenizer text #########################################
                with torch.no_grad():
                    encoded_input = tokenizer(data['plot'], return_tensors='pt', truncation=True)
                    encoded_input.to(device)
                ##################################### load image #####################################
                if type(im_file) == str:
                    image = VGGClassifier.resize_and_crop_image(im_file)
                with torch.no_grad():
                    vgg_feature=torch.from_numpy(image).to(device)    
                    #print(input_image.shape)
            
    ##################################### prediction ##################################### 
    with torch.no_grad():
        if options == "encoder" or options == "normal":
            out = model([encoded_input,vgg_feature],options)
        else:
            x = []
            out = model(x,options)


options == "normal"
with torch.no_grad():
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            './log/mmimdb_agg'),
        record_shapes=True,
        with_stack=True)
    with prof as p: 
        for i, file in enumerate(files):
            with open(file) as f:
                data = json.load(f)
                data['imdb_id'] = file.split('/')[-1].split('.')[0]
                # if 'plot' in data and 'plot outline' in data:
                #    data['plot'].append(data['plot outline'])
                im_file = file.replace('json', 'jpeg')
                if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):
                    plot_id = numpy.array([len(p) for p in data['plot']]).argmax()
                    
                    data['plot'] = data['plot'][plot_id]

                    ##################################### tokenizer text #########################################
                    with torch.no_grad():
                        encoded_input = tokenizer(data['plot'], return_tensors='pt', truncation=True)
                        encoded_input.to(device)
                    ##################################### load image #####################################
                    if type(im_file) == str:
                        image = VGGClassifier.resize_and_crop_image(im_file)
                    with torch.no_grad():
                        vgg_feature=torch.from_numpy(image).to(device)    
                        #print(input_image.shape)
                    
            ##################################### prediction ##################################### 
            with torch.no_grad():
                    out = model([encoded_input,vgg_feature],options)
            p.step()