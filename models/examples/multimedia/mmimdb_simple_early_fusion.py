import torch
import sys
import os
import numpy as np
import random
from PIL import Image
sys.path.append(os.getcwd())

from unimodals.common_models import MaxOut_MLP, Identity
#from datasets.imdb.get_data_mmimdb_simple_early_fusion import get_dataloader
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test

import torchvision.models as models
from torch.nn import Linear
import json
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
filename = "best_ef.pt"
print("start load!")
#"/home/xh/20220601_mmbench/MultiBench/datasets/imdb/gmu-mmimdb/multimodal_imdb.hdf5"




traindata, validdata, testdata = get_dataloader(
    device,"/home/xh/20220601_mmbench/MultiBench/datasets/imdb/gmu-mmimdb/multimodal_imdb_back.hdf5", "/home/xh/benchdata/Multimedia/mmimdb", vgg=True, batch_size=128, no_robust=True,num_workers=0)
print("finish load!")
#sys.exit(-1)
#sys.path.append('/home/xh/20220601_mmbench/MultiBench/datasets/imdb/gmu-mmimdb')

conf_file = sys.argv[1] if len(sys.argv) > 1 else None
with open(conf_file) as f:
    locals().update(json.load(f))
class VGGClassifier(object):

    def __init__(self,model_path='vgg.tar', synset_words='synset_words.txt'):
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.add_module("add_linear",Linear(1000,4096))
        print(self.vgg16)
        self.vgg16.eval()
        self.vgg16=self.vgg16.to(device)
        pass
    def get_features(self, image):
        """Returns the activations of the last hidden layer for a given image.

        :image: numpy image or image path.
        :returns: numpy vector with 4096 activations.

        """
        if type(image) == str:
            image = VGGClassifier.resize_and_crop_image(image)
        with torch.no_grad():
            input_image=torch.from_numpy(image).to(device)    
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
#from transformers import AlbertTokenizer, AlbertModel


#model_transformer.eval()
#model_transformer.to(device)
"""
from vgg_modify import VGGClassifier
image_encoder=VGGClassifier()
image_encoder.vgg16.add_module("add_linear",Linear(1000,4096))
image_encoder.vgg16.eval()
#image_encoder.vgg16.to(device)
#encoders = [Identity(), Identity()]
"""
encoders = [Identity(), Identity() ]
head = MaxOut_MLP(23, 512, 4396).to(device)







#head = MaxOut_MLP(23, 512, 4864).to(device)




"""
fusion = Concat().to(device)

train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel",
      save=filename, optimtype=torch.optim.AdamW, lr=4e-2, weight_decay=0.01, objective=torch.nn.BCEWithLogitsLoss())
"""


print("Testing:")
model = torch.load(filename).to(device)
test(model, testdata, method_name="ef", dataset="imdb",
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel", no_robust=True)


