
import numpy


from PIL import Image

import torchvision.models as models
import torch
import time
from torch.nn import Linear
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)



class VGGClassifier(object):

    def __init__(self, model_path='vgg.tar', synset_words='synset_words.txt'):
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
