import torch
import sys
import os
import numpy as np
import random
from PIL import Image
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
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.add_module("add_linear",Linear(1000,4096))
        print(self.vgg16)
        self.vgg16.eval()
        self.vgg16=self.vgg16.to(device)
        self.device=device
        pass
    def get_features(self, image):
        """Returns the activations of the last hidden layer for a given image.

        :image: numpy image or image path.
        :returns: numpy vector with 4096 activations.

        """
        if type(image) == str:
            image = VGGClassifier.resize_and_crop_image(image)
        with torch.no_grad():
            input_image=torch.from_numpy(image).to(self.device)    
            #print(input_image.shape)
            acti_layer=self.vgg16(input_image) 
            acti_layer=self.vgg16.add_linear(acti_layer) 
            
            acti_layer=acti_layer.cpu()
            #print(acti_layer)
            del input_image
            del image
            #time.sleep(0.1)
            
            torch.cuda.empty_cache()
            

            #print(acti_layer.shape)
            #print(acti_layer.shape)
            return acti_layer
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
        
        ### modules
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.add_module("add_linear",Linear(1000,4096))


        #print(self.vgg16)

        #self.vgg16.eval()
        #self.vgg16=self.vgg16.to(device)
    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """

        outs = []
        acti_layer=self.vgg16(inputs[1]) 
        inputs[1]=self.vgg16.add_linear(acti_layer) 
            
        for i in range(len(inputs)):
            outs.append(self.encoders[i](inputs[i]))
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
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
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


from transformers import AlbertTokenizer, AlbertModel
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


encoders = [Identity(), Identity() ]
head = MaxOut_MLP(23, 512, 4864).to(device)
fusion = Concat().to(device)
model = MMDL(encoders, fusion, head, has_padding=False).to(device)
model.load_state_dict(torch.load("best_ef5489_dict.pth"))
model.eval()
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
            #print("data['plot'][plot_id]:",data['plot'][plot_id])
            #print("normalizeText:",normalizeText(data['plot'][plot_id]))
            data['plot_back']=normalizeText(data['plot'][plot_id])
            
            
            data['plot'] = data['plot'][plot_id]
            #print("type:",type(data['plot']))
            #print("shape:",data['plot'].shape)

            ##################################### handel text #########################################
            with torch.no_grad():
                encoded_input = tokenizer(data['plot'], return_tensors='pt', truncation=True)
                encoded_input.to(device)
                
                #tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                
                encoded_output=model_transformer(**encoded_input)
                encoded_output=encoded_output.last_hidden_state[:,-1,:].cpu()
                del encoded_input
            ##################################### end handel text #####################################
            #print("type:",type(data['plot']))

            """"""


            if len(data['plot_back']) > 0:
                vocab_counts.extend(data['plot_back'])
                #print("im_file:",im_file,img_size)
                data['cover'] = VGGClassifier.resize_and_crop_image(
                    im_file, img_size)   ###
                #self.dataset["images"].append(data['cover'])


                ##################################### handel image #########################################
                
                #self.dataset["images"].append(copy.deepcopy(data['vgg_features']))
                #data['vgg_features'] = clsf.get_features(im_file) ###vgg16
                if type(im_file) == str:
                    image = VGGClassifier.resize_and_crop_image(image)
                with torch.no_grad():
                    vgg_feature=torch.from_numpy(image).to(device)    
                    #print(input_image.shape)
                    

                ##################################### end handel image #####################################

            

    ##################################### prediction ##################################### 
    with torch.no_grad():
        

        #print("input:",encoded_output)
        #print()
        #print(vgg_feature)
        
        out = model([_processinput(x).float().to(device)
                    for x in [encoded_output,vgg_feature]])
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


