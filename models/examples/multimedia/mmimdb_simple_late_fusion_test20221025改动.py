import torch
import sys
sys.path.append(os.getcwd())
import os
import numpy as np
import random
from PIL import Image
import argparse
from torch import nn
softmax = nn.Softmax()
import torchvision.models as models
from torch.nn import Linear
import json
import sklearn.metrics
'''

                            _ooOoo_
                           o8888888o
                           88" . "88
                           (| -_- |)
                            O\ = /O
                        ____/`---'\____
                      .   ' \\| |// `.
                       / \\||| : |||// \
                     / _||||| -:- |||||- \
                       | | \\\ - /// | |
                     | \_| ''\---/'' | |
                      \ .-\__ `-` ___/-. /
                   ___`. .' /--.--\ `. . __
                ."" '< `.___\_<|>_/___.' >'"".
               | | : `- \`.;`\ _ /`;.`/ - ` : | |
                 \ \ `-. \_ __\ /__ _/ .-` / /
         ======`-.____`-.___\_____/___.-`____.-'======
                            `=---='

         .............................................
                  Buddha bless, Bugless

'''
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model_name', help='name of model')
    parser.add_argument('--gpu', type=int, default=0, help='name of model')
    parser.add_argument('--seed', type=int, default=114514, help='name of model')
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

if "mmimdb" in args.model_name:
if args.model_name=="mmimdb_simple_late_fusion":
    encoders = [MaxOut_MLP(512, 512, 768, linear_layer=False),
                MaxOut_MLP(512, 1024, 4096, 512, False)]
    head = Linear(1024, 23).to(device)
    fusion = Concat().to(device)

elif args.model_name=="mmimdb_simple_early_fusion":
    ### mmimdb_simple_early_fusion
    from unimodals.common_models import Identity
    encoders = [Identity(), Identity() ]
    head = MaxOut_MLP(23, 512, 4864).to(device)
    fusion = Concat().to(device)

elif args.model_name=="mmimdb_low_rank_tensor":
    ###mmimdb_low_rank_tensor
    from fusions.common_fusions import LowRankTensorFusion
    encoders = [MaxOut_MLP(512, 512, 768, linear_layer=False),
                MaxOut_MLP(512, 1024, 4096, 512, False)]
    head = Linear(512, 23).to(device)
    fusion = LowRankTensorFusion([512, 512], 512, 128).to(device)
    
elif args.model_name=="mmimdb_multi_interac_matrix":
    from fusions.common_fusions import MultiplicativeInteractions2Modal
    encoders = [MaxOut_MLP(512, 512, 768, linear_layer=False),
                MaxOut_MLP(512, 1024, 4096, 512, False)]
    head = Linear(1024, 23).cuda()
    fusion = MultiplicativeInteractions2Modal([512, 512], 1024, 'matrix').cuda()

elif args.model_name=="mmimdb_low_rank_tensor":
    encoders = [MaxOut_MLP(512, 512, 768, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 512, False)]
    head = Linear(512, 23).cuda()
    fusion = LowRankTensorFusion([512, 512], 512, 128).cuda()

from unimodals.common_models import MaxOut_MLP
from unimodals.common_models import Linear as mod_linear
from fusions.common_fusions import Concat
#from training_structures.Supervised_Learning import train,test
from transformers import AlbertTokenizer, AlbertModel

if "mmimdb" in args.model_name:
    pass
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
        encoded_output=self.encoders[0].MaxOut_MLP0(encoded_output)
        outs.append(encoded_output)
        #acti_layer=self.encoders[1](inputs[1])
        #print("inputs[1]",inputs[1])
        acti_layer=self.encoders[1](inputs[1].float())#
        acti_layer=self.encoders[1].add_linear(acti_layer).float()
        acti_layer=self.encoders[1].MaxOut_MLP1(acti_layer)
        

        #for param_tensor in vgg16.state_dict():
        #    print(param_tensor, "\t", vgg16.state_dict()[param_tensor])
        #print("acti_layer",acti_layer)
        outs.append(acti_layer)
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


labels=np.load("imdb_res.npy")


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


with open("mapping", "rb") as fp:   # Unpickling
    mapping = pickle.load(fp)
    #print(b)

with open('list.txt', 'r') as f:
    #files = f.read().splitlines()[18160:25959] ## the test set of mmimdf
    files_origin = f.read().splitlines()#[25859:25959]
files=[]
for i in range(0,25959):
    files.append(files_origin[mapping[i][1]])
#files=files[18160:25959]## the test set
files=files[25859:25959]
## Load data and define vocab ##
logger.info('Reading json and jpeg files...')

vocab_counts = []

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model_transformer = AlbertModel.from_pretrained("albert-base-v2")

model_transformer .add_module("MaxOut_MLP0",MaxOut_MLP(512, 512, 768, linear_layer=False))

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
vgg16.add_module("MaxOut_MLP1",MaxOut_MLP(512, 1024, 4096, 512, False))
#print(vgg16.MaxOut_MLP1.op1.lin.weight)
#sys.exit(-1)
vgg16.eval()
#encoders = [Identity(), Identity() ]
encoders = [model_transformer,vgg16 ]
head = mod_linear(1024, 23).to(device)

fusion = Concat().to(device)

model = MMDL(encoders, fusion, head, has_padding=False).to(device)
#model.load_state_dict(torch.load("best_ef5489_dict.pth"))

model.eval()
"""

model.encoders=nn.ModuleList([model_transformer,vgg16  ])
torch.save(model.state_dict(), "best_ef5489_dict_end_to_end.pth")
print("saved!!")
model.to(device)
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print(model)

model.eval()
"""
#sys.exit(-1)
############################## load image/text  ##############################################
for i, file in enumerate(files):
    logger.info('{0:05d} out of {1:05d}: {2:02.2f}%'.format(
        i, len(files), float(i) / len(files) * 100))    
    #single_test
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
        import torch
        import torch.nn as nn
        from torch.autograd import Variable
        from functools import reduce
        import operator
        from layers import LearnedGroupConv, CondensingLinear, CondensingConv, Conv


        

        #sys.exit(-1)

        #print("input:",encoded_output)
        #print()
        #print(vgg_feature)
        
        out = model([encoded_input,vgg_feature])
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
true=np.array(labels[-100:])
#true=np.array(labels)
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





