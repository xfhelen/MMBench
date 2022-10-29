import torch
import sys
import os
import numpy as np
import random
from PIL import Image
from thop import profile
sys.path.append(os.getcwd())

from unimodals.common_models import MaxOut_MLP, Identity
#from datasets.imdb.get_data_mmimdb_simple_early_fusion import get_dataloader
#from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
#from training_structures.Supervised_Learning import train,test
from torch import nn
softmax = nn.Softmax()
import torchvision.models as models
from torch.nn import Linear
import json
import sklearn.metrics
from transformers import AlbertTokenizer, AlbertModel


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)




os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
filename = "best_ef5489.pt"

#"/home/xh/20220601_mmbench/MultiBench/datasets/imdb/gmu-mmimdb/multimodal_imdb.hdf5"






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
        #self.model_transformer = AlbertModel.from_pretrained("albert-base-v2")
        #self.model_transformer.to(device)
        #self.model_transformer.eval()
    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        
        outs = []
        
        
        #encoded_output=model_transformer(**inputs[0])
        encoded_output=self.encoders[0](**inputs[0])
        encoded_output=encoded_output.last_hidden_state[:,-1,:]#.float()
        outs.append(encoded_output)
        #acti_layer=self.encoders[1](inputs[1])
        #print("inputs[1]",inputs[1])
        acti_layer=self.encoders[1](inputs[1].float())#

        

        #for param_tensor in vgg16.state_dict():
        #    print(param_tensor, "\t", vgg16.state_dict()[param_tensor])
        #print("acti_layer",acti_layer)
        outs.append(self.encoders[1].add_linear(acti_layer).float())
        #print("outs[0]",outs[0])
        #print("outs[1]",outs[1])
        
        """
        for i in range(len(inputs)):
            outs.append(self.encoders[i](inputs[i]))
        """
        self.reps = outs
        if self.has_padding:
            
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        #print("self.fuse(outs)",out)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        #print("out = out[0]1",out)
        
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        #print("out = out[0]2",head(out))
        #sys.exit(-1)
        return self.head(out)


"""
import h5py
dataset = h5py.File("/home/xh/20220601_mmbench/MultiBench/datasets/imdb/gmu-mmimdb/multimodal_imdb.hdf5", 'r')
print(dataset["genres"][18160])
print(type(dataset["genres"][18160]))
label_list=[]
for i in range(18160,25959):
    label_list.append(dataset["genres"][i])
label_list=np.array(label_list)
np.save("imdb_res.npy", label_list)
"""


############################## init parameters ##############################################
import numpy
import re
import sys
#from robustness.text_robust import add_text_noise
#from robustness.visual_robust import add_visual_noise
import os
import sys
from typing import *
import numpy as np
import json
import logging
import math
#from models.eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
#from models.eval_scripts.complexity import all_in_one_train, all_in_one_test
#from models.eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
import copy
import pickle
load_in_memory=True
batch_size=128
model_name="gmu"
model_class="GatedTrainer"
sources=['genres', 'vgg_features', 'features']
hidden_size=512
learning_rate=0.01
init_ranges=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
max_norms=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
img_size=[160, 256]
num_channels=3
threshold=0.5
n_epochs=200
dropout=0.5
test_size=0.3
dev_size=0.1
word2vec_path="/home/xh/benchdata/Multimedia/mmimdb/GoogleNews-vectors-negative300.bin.gz"
rng_seed=[2014, 8, 6]
n_classes=23
textual_dim=768
visual_dim=4096
labels=np.load("imdb_res.npy")


criterion=torch.nn.BCEWithLogitsLoss()
task="multilabel"
auprc=False
input_to_float=True
no_robust=True
"""
print(labels)
print(labels.shape)
sys.exit(-1)

print("start load!")
traindata, validdata, testdata = get_dataloader(
    device,"/home/xh/20220601_mmbench/MultiBench/datasets/imdb/gmu-mmimdb/multimodal_imdb_back.hdf5", "/home/xh/benchdata/Multimedia/mmimdb", vgg=True, batch_size=128, no_robust=True,num_workers=0)
print("finish load!")
"""


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


with open("mapping", "rb") as fp:   # Unpickling
    mapping = pickle.load(fp)
    #print(b)

with open('list.txt', 'r') as f:
    #files = f.read().splitlines()[18160:25959] ## the test set of mmimdf
    files_origin = f.read().splitlines()#[25859:25959]
files=[]
for i in range(0,25959):
    files.append(files_origin[mapping[i][1]])
files=files[18160:25959]## the test set
#files=files[25859:25959]
## Load data and define vocab ##
logger.info('Reading json and jpeg files...')

vocab_counts = []



tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model_transformer = AlbertModel.from_pretrained("albert-base-v2")

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

totalloss = 0.0
pred = []
true = []

vgg16 = models.vgg16(pretrained=True)
vgg16.add_module("add_linear",Linear(1000,4096))
vgg16.eval()
encoders = [Identity(), Identity() ]
#encoders = [model_transformer,vgg16  ]
head = MaxOut_MLP(23, 512, 4864).to(device)
fusion = Concat().to(device)
model = MMDL(encoders, fusion, head, has_padding=False).to(device)

model.load_state_dict(torch.load("best_ef5489_dict.pth"))

"""
model.load_state_dict(torch.load("best_ef5489_dict_end_to_end.pth"))
model.eval()
"""


#model = torch.load(filename).to(device)
model.encoders=nn.ModuleList([model_transformer,vgg16  ])
torch.save(model.state_dict(), "best_ef5489_dict_end_to_end.pth")

print("saved!!")
model.to(device)
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print(model)

model.eval()

#sys.exit(-1)
############################## load image/text  ##############################################
for i, file in enumerate(files):
    #logger.info('{0:05d} out of {1:05d}: {2:02.2f}%'.format(
        #i, len(files), float(i) / len(files) * 100))    
    #single_test
    with open(file) as f:
        data = json.load(f)
        data['imdb_id'] = file.split('/')[-1].split('.')[0]
        # if 'plot' in data and 'plot outline' in data:
        #    data['plot'].append(data['plot outline'])
        im_file = file.replace('json', 'jpeg')
        if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):
            plot_id = numpy.array([len(p) for p in data['plot']]).argmax()
            #print("data['plot'][plot_id]:",data['plot'][plot_id])
            #print("normalizeText:",normalizeText(data['plot'][plot_id]))
            #data['plot_back']=normalizeText(data['plot'][plot_id])
            
            
            data['plot'] = data['plot'][plot_id]
            #print("type:",type(data['plot']))
            #print("shape:",data['plot'].shape)

            ##################################### tokenizer text #########################################
            with torch.no_grad():
                encoded_input = tokenizer(data['plot'], return_tensors='pt', truncation=True)
                encoded_input.to(device)
                
                #print("encoded_input1:",type(encoded_input))
                #print("encoded_input2:",encoded_input)
                
            ##################################### end tokenizer text #####################################
            #print("type:",type(data['plot']))

            
            ##################################### load image #####################################
            if type(im_file) == str:
                image = VGGClassifier.resize_and_crop_image(im_file)
            with torch.no_grad():
                vgg_feature=torch.from_numpy(image).to(device)
                
                #print(input_image.shape)
            ##################################### end load image ################################# 
            
                

            

    ##################################### prediction ##################################### 
    with torch.no_grad():
        

        #print("input:",encoded_output)
        #print()
        #print(vgg_feature)
        
        out = model([encoded_input,vgg_feature])
        print(out)
        #print("out")
        flops, params =profile(model, inputs=[encoded_input,vgg_feature])
        print(params,flops)
        #flops, params =profile(model_transformer, inputs= (**encoded_input,))
        flops, params =profile(vgg16, inputs= vgg_feature)
        print(params,flops)






        from typing import Union, List, Dict, Any, cast
        from torch.nn import Linear
        import torch.nn as nn
        __all__ = [
            'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
            'vgg19_bn', 'vgg19',
        ]


        model_urls = {
            'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
            'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
            'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
            'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
            'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
            'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
            'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
        }


        class VGG(nn.Module):

            def __init__(
                self,
                features: nn.Module,
                num_classes: int = 1000,
                init_weights: bool = True
            ) -> None:
                super(VGG, self).__init__()
                self.features = features
                self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes),
                )
                if init_weights:
                    self._initialize_weights()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

            def _initialize_weights(self) -> None:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)


        def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
            layers: List[nn.Module] = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    v = cast(int, v)
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers)


        cfgs: Dict[str, List[Union[str, int]]] = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }


        def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
            if pretrained:
                kwargs['init_weights'] = False
            model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
            
            return model




        def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
            r"""VGG 16-layer model (configuration "D")
            `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
            The required minimum input size of the model is 32x32.

            Args:
                pretrained (bool): If True, returns a model pre-trained on ImageNet
                progress (bool): If True, displays a progress bar of the download to stderr
            """
            
            return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


        vgg16model = VGG(make_layers(cfgs["D"], batch_norm=False))
        vgg16model.add_module("add_linear",Linear(1000,4096))
        vgg16model.to(device)
        vgg16model.eval()
        flops, params =profile(vgg16model, inputs= vgg_feature)
        print(params,flops)
        sys.exit(-1)
        if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
            loss = criterion(out[0], torch.from_numpy(labels[i]).float().to(device))

        #print("out:",out)
        

        
        totalloss += loss*len(labels[i])
        
        #.cpu().numpy()
        pred.append(torch.sigmoid(out[0]).round().cpu().numpy())
        #print("pred:",pred[-1])
        #print("label:",labels[i])
        true.append(torch.from_numpy(labels[i]).to(device) )
    ##################################### end prediction #################################
"""
if pred:
    pred = torch.cat(pred, 0)
print(type(true))
print(true[0])
print(type(pred))
print(pred[0])
true = torch.cat(true, 0)
print(type(true))
print(true[0])
""" 
pred=np.array(pred)
#true=np.array(labels[-100:])
true=np.array(labels)
totals = len(true)#.shape[0]
testloss = totalloss/totals
print("shape1:",pred.shape)
print("shape2:",true.shape)

print(" f1_micro: "+str(sklearn.metrics.f1_score(true, pred, average="micro")  ) +
        " f1_macro: "+str(sklearn.metrics.f1_score(true, pred, average="macro") ))
 
"""
print(" f1_micro: "+str(f1_score(true[:100], pred, average="micro")) +
        " f1_macro: "+str(f1_score(true[:100], pred, average="macro")))
"""

#test(model, testdata, method_name="ef", dataset="imdb",
     #criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel", no_robust=True)


