"""Implements dataloaders for IMDB dataset."""

from tqdm import tqdm
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import h5py
from gensim.models import KeyedVectors


from robustness.text_robust import add_text_noise
from robustness.visual_robust import add_visual_noise
import os
import sys
from typing import *
import numpy as np
import json
import logging
import math
import os
import numpy
import re
import sys
from collections import OrderedDict, Counter
#from fuel.datasets import H5PYDataset
#from fuel.utils import find_in_data_path

from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from matplotlib import pyplot
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

import torchvision.models as models
import time
from torch.nn import Linear
import torch 
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

sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')

class VGGClassifier(object):

    def __init__(self,device, model_path='vgg.tar', synset_words='synset_words.txt'):
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
class data_origin():
    def __init__(self,device) -> None:
        """Initialize IMDBDataset object.

        Args:
            file (h5py.File): h5py file of data
            start_ind (int): Starting index for dataset
            end_ind (int): Ending index for dataset
            vggfeature (bool, optional): Whether to return pre-processed vgg_features or not. Defaults to False.
        
        
        self.dataset={}
        self.dataset["features"]=[]
        self.dataset["images"]=[]
        self.dataset["genres"]=[]
        """
        self.device=device


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

        

        with open('/home/xucheng/xh/MultiBench/list.txt', 'r') as f:
            files = f.read().splitlines()

        ## Load data and define vocab ##
        logger.info('Reading json and jpeg files...')
        movies = []
        vocab_counts = []


        from transformers import AlbertTokenizer, AlbertModel
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model_transformer = AlbertModel.from_pretrained("albert-base-v2")
        model_transformer.to(device)
        model_transformer.eval()
        """"""
        clsf = VGGClassifier(device)
        for i, file in enumerate(files):
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
                    with torch.no_grad():
                        encoded_input = tokenizer(data['plot'], return_tensors='pt', truncation=True)
                        encoded_input.to(device)
                        
                        #tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                        
                        encoded_output=model_transformer(**encoded_input)
                        encoded_output=encoded_output.last_hidden_state[:,-1,:].cpu()
                        del encoded_input
                    #print("type:",type(data['plot']))
                    #print("shape:",data['plot'].shape)
                    
                    #self.dataset['features'].append(copy.deepcopy(encoded_output))

                    """
                    if len(data['plot_back'])  >512:
                        data['plot_back']=data['plot_back'][:512]
                        data['plot'] =""
                        for tmp_i in range(512):
                            data['plot']=data['plot']+data['plot_back'][tmp_i]+" "
                    else:
                        data['plot'] = data['plot'][plot_id]
                    """
                    
                    #data['plot'] = normalizeText(data['plot'][plot_id])

                    """"""


                    if len(data['plot_back']) > 0:
                        vocab_counts.extend(data['plot_back'])
                        print("im_file:",im_file,img_size)
                        data['cover'] = VGGClassifier.resize_and_crop_image(
                            im_file, img_size)   ###
                        #self.dataset["images"].append(data['cover'])

                        data['vgg_features'] = clsf.get_features(im_file)
                        
                        #self.dataset["images"].append(copy.deepcopy(data['vgg_features']))
                        movies.append(data)
                    
            logger.info('{0:05d} out of {1:05d}: {2:02.2f}%'.format(
                i, len(files), float(i) / len(files) * 100))

        logger.info('done reading files.')

           
        vocab_counts = OrderedDict(Counter(vocab_counts).most_common())
        vocab = ['_UNK_'] + [v for v in vocab_counts.keys()]
        #googleword2vec =KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        ix_to_word = dict(zip(range(len(vocab)), vocab))
        word_to_ix = dict(zip(vocab, range(len(vocab))))
        
        """
        lookup = numpy.array([googleword2vec[v] for v in vocab if v in googleword2vec])
        print("lookup:",lookup)
        numpy.save('metadata.npy', {'ix_to_word': ix_to_word,
                                    'word_to_ix': word_to_ix,
                                    'vocab_size': len(vocab),
                                    'lookup': lookup})
        """


        numpy.save('metadata.npy', {'ix_to_word': ix_to_word,
                                    'word_to_ix': word_to_ix,
                                    'vocab_size': len(vocab)})
        # Define train, dev and test subsets
        counts = OrderedDict(
            Counter([g for m in movies for g in m['genres']]).most_common())
        target_names = list(counts.keys())[:n_classes]

        le = MultiLabelBinarizer()
        Y = le.fit_transform([m['genres'] for m in movies])
        labels = numpy.nonzero(le.transform([[t] for t in target_names]))[1]

        B = numpy.copy(Y)
        rng = numpy.random.RandomState(rng_seed)
        train_idx, dev_idx, test_idx = [], [], []
        for l in labels[::-1]:
            t = B[:, l].nonzero()[0]
            t = rng.permutation(t)
            n_test = int(math.ceil(len(t) * test_size))
            n_dev = int(math.ceil(len(t) * dev_size))
            n_train = len(t) - n_test - n_dev
            test_idx.extend(t[:n_test])
            dev_idx.extend(t[n_test:n_test + n_dev])
            train_idx.extend(t[n_test + n_dev:])
            B[t, :] = 0

        indices = numpy.concatenate([train_idx, dev_idx, test_idx])

        #self.dataset["genres"]= Y[indices][:, labels]

        
        nsamples = len(indices)
        print("nsamples:",nsamples)
        nsamples_train, nsamples_dev, nsamples_test = len(
            train_idx), len(dev_idx), len(test_idx)

        ## use transformers to replace googleword2vec
        from transformers import AlbertTokenizer, AlbertModel
        import numpy as np


        #model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        #word2vec_pipeline=pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=2)
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained("albert-base-v2")
        print(type(model))
        model.to(device)
        #sys.exit(-1)
        # Obtain feature vectors and text sequences
        sequences = []
        X = numpy.zeros((indices.shape[0], textual_dim), dtype='float32')
        for i, idx in enumerate(indices):
            words = movies[idx]['plot']
            #print(movies[idx]['plot'])
            #print(movies[idx]['plot_back'])
            #print(len(movies[idx]['plot_back']))
            sequences.append([word_to_ix[w] if w in vocab else unk_idx for w in movies[idx]['plot_back']])

            #X[i] = numpy.array([googleword2vec[w]
                                #for w in words if w in googleword2vec]).mean(axis=0)
            encoded_input = tokenizer(words, return_tensors='pt', truncation=True)
            encoded_input.to(device)
            output = model(**encoded_input)
            #print("type(output):",type(output))
            #print("output.shape:",output.last_hidden_state[:,-1,:].size())
            output=output.last_hidden_state[:,-1,:].cpu().detach().numpy()
            
            
            flat_output=output.ravel()
            #print(flat_output)
            #print(type(flat_output))
            #print(flat_output.shape)
            if flat_output.shape[0]<768:
                flat_output=np.concatenate((flat_output,np.zeros(  768-flat_output.shape[0])))
            else:
                flat_output=flat_output[:768]
            
            X[i] =flat_output
            #print("X[i].shape) shape:300
        #del googleword2vec

        # get n-grams representation
        sentences = [' '.join(m['plot_back']) for m in movies]
        ngram_vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=(3, 3), min_df=2)
        ngrams_feats = ngram_vectorizer.fit_transform(sentences).astype('float32')
        word_vectorizer = TfidfVectorizer(min_df=10)
        wordgrams_feats = word_vectorizer.fit_transform(sentences).astype('float32')

        
        # Store data in the hdf5 file
        f = h5py.File('multimodal_imdb.hdf5', mode='w')
        dtype = h5py.special_dtype(vlen=numpy.dtype('int32'))
        print("X.shape:",X.shape)
        print("Y.shape:",Y.shape)
        features = f.create_dataset('features', X.shape, dtype='float32')
        vgg_features = f.create_dataset(
            'vgg_features', (nsamples, 4096), dtype='float32')
        three_grams = f.create_dataset(
            'three_grams', (nsamples, ngrams_feats.shape[1]), dtype='float32')
        word_grams = f.create_dataset(
            'word_grams', (nsamples, wordgrams_feats.shape[1]), dtype='float32')
        images = f.create_dataset(
            'images', [nsamples, num_channels] + img_size[::-1], dtype='int32')
        seqs = f.create_dataset('sequences', (nsamples,), dtype=dtype)
        genres = f.create_dataset('genres', (nsamples, n_classes), dtype='int32')
        imdb_ids = f.create_dataset('imdb_ids', (nsamples,), dtype="S7")
        imdb_ids[...] = numpy.asarray([m['imdb_id']
                                    for m in movies], dtype='S7')[indices]
        features[...] = X
        for i, idx in enumerate(indices):
            images[i] = movies[idx]['cover']
            vgg_features[i] = movies[idx]['vgg_features']
        seqs[...] = sequences
        genres[...] = Y[indices][:, labels]
        
        
        three_grams[...] = ngrams_feats[indices].todense()
        word_grams[...] = wordgrams_feats[indices].todense()
        genres.attrs['target_names'] = json.dumps(target_names)
        features.dims[0].label = 'batch'
        features.dims[1].label = 'features'
        three_grams.dims[0].label = 'batch'
        three_grams.dims[1].label = 'features'
        word_grams.dims[0].label = 'batch'
        word_grams.dims[1].label = 'features'
        imdb_ids.dims[0].label = 'batch'
        genres.dims[0].label = 'batch'
        genres.dims[1].label = 'classes'
        vgg_features.dims[0].label = 'batch'
        vgg_features.dims[1].label = 'features'
        images.dims[0].label = 'batch'
        images.dims[1].label = 'channel'
        images.dims[2].label = 'height'
        images.dims[3].label = 'width'

        split_dict = {
            'train': {
                'features': (0, nsamples_train),
                'three_grams': (0, nsamples_train),
                'sequences': (0, nsamples_train),
                'images': (0, nsamples_train),
                'vgg_features': (0, nsamples_train),
                'imdb_ids': (0, nsamples_train),
                'word_grams': (0, nsamples_train),
                'genres': (0, nsamples_train)},
            'dev': {
                'features': (nsamples_train, nsamples_train + nsamples_dev),
                'three_grams': (nsamples_train, nsamples_train + nsamples_dev),
                'sequences': (nsamples_train, nsamples_train + nsamples_dev),
                'images': (nsamples_train, nsamples_train + nsamples_dev),
                'vgg_features': (nsamples_train, nsamples_train + nsamples_dev),
                'imdb_ids': (nsamples_train, nsamples_train + nsamples_dev),
                'word_grams': (nsamples_train, nsamples_train + nsamples_dev),
                'genres': (nsamples_train, nsamples_train + nsamples_dev)},
            'test': {
                'features': (nsamples_train + nsamples_dev, nsamples),
                'three_grams': (nsamples_train + nsamples_dev, nsamples),
                'sequences': (nsamples_train + nsamples_dev, nsamples),
                'images': (nsamples_train + nsamples_dev, nsamples),
                'vgg_features': (nsamples_train + nsamples_dev, nsamples),
                'imdb_ids': (nsamples_train + nsamples_dev, nsamples),
                'word_grams': (nsamples_train + nsamples_dev, nsamples),
                'genres': (nsamples_train + nsamples_dev, nsamples)}
        }

        #f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        self.dataset=f
        #f.flush()
        #f.close()
        """"""

class IMDBDataset(Dataset):
    """Implements a torch Dataset class for the imdb dataset."""
    
    def __init__(self, device,file: h5py.File, start_ind: int, end_ind: int, vggfeature: bool = False) -> None:
        


        """    
        vocab_counts = OrderedDict(Counter(vocab_counts).most_common())
        vocab = ['_UNK_'] + [v for v in vocab_counts.keys()]
        #googleword2vec =KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        ix_to_word = dict(zip(range(len(vocab)), vocab))
        word_to_ix = dict(zip(vocab, range(len(vocab))))
        
        lookup = numpy.array([googleword2vec[v] for v in vocab if v in googleword2vec])
        print("lookup:",lookup)
        numpy.save('metadata.npy', {'ix_to_word': ix_to_word,
                                    'word_to_ix': word_to_ix,
                                    'vocab_size': len(vocab),
                                    'lookup': lookup})
        


        numpy.save('metadata.npy', {'ix_to_word': ix_to_word,
                                    'word_to_ix': word_to_ix,
                                    'vocab_size': len(vocab)})
        # Define train, dev and test subsets
        counts = OrderedDict(
            Counter([g for m in movies for g in m['genres']]).most_common())
        target_names = list(counts.keys())[:n_classes]

        le = MultiLabelBinarizer()
        Y = le.fit_transform([m['genres'] for m in movies])
        labels = numpy.nonzero(le.transform([[t] for t in target_names]))[1]

        B = numpy.copy(Y)
        rng = numpy.random.RandomState(rng_seed)
        train_idx, dev_idx, test_idx = [], [], []
        for l in labels[::-1]:
            t = B[:, l].nonzero()[0]
            t = rng.permutation(t)
            n_test = int(math.ceil(len(t) * test_size))
            n_dev = int(math.ceil(len(t) * dev_size))
            n_train = len(t) - n_test - n_dev
            test_idx.extend(t[:n_test])
            dev_idx.extend(t[n_test:n_test + n_dev])
            train_idx.extend(t[n_test + n_dev:])
            B[t, :] = 0

        indices = numpy.concatenate([train_idx, dev_idx, test_idx])
        nsamples = len(indices)
        print("nsamples:",nsamples)
        nsamples_train, nsamples_dev, nsamples_test = len(
            train_idx), len(dev_idx), len(test_idx)

        ## use transformers to replace googleword2vec
        from transformers import AlbertTokenizer, AlbertModel
        import numpy as np


        #model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        #word2vec_pipeline=pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=2)
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained("albert-base-v2")
        print(type(model))
        model.to(device)
        #sys.exit(-1)
        # Obtain feature vectors and text sequences
        sequences = []
        X = numpy.zeros((indices.shape[0], textual_dim), dtype='float32')
        for i, idx in enumerate(indices):
            words = movies[idx]['plot']
            #print(movies[idx]['plot'])
            #print(movies[idx]['plot_back'])
            #print(len(movies[idx]['plot_back']))
            sequences.append([word_to_ix[w] if w in vocab else unk_idx for w in movies[idx]['plot_back']])

            #X[i] = numpy.array([googleword2vec[w]
                                #for w in words if w in googleword2vec]).mean(axis=0)
            encoded_input = tokenizer(words, return_tensors='pt', truncation=True)
            encoded_input.to(device)
            output = model(**encoded_input)
            #print("type(output):",type(output))
            #print("output.shape:",output.last_hidden_state[:,-1,:].size())
            output=output.last_hidden_state[:,-1,:].cpu().detach().numpy()
            
            
            flat_output=output.ravel()
            #print(flat_output)
            print(type(flat_output))
            print(flat_output.shape)
            if flat_output.shape[0]<768:
                flat_output=np.concatenate((flat_output,np.zeros(  768-flat_output.shape[0])))
            else:
                flat_output=flat_output[:768]
            
            X[i] =flat_output
            #print("X[i].shape) shape:300
        #del googleword2vec

        # get n-grams representation
        sentences = [' '.join(m['plot_back']) for m in movies]
        ngram_vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=(3, 3), min_df=2)
        ngrams_feats = ngram_vectorizer.fit_transform(sentences).astype('float32')
        word_vectorizer = TfidfVectorizer(min_df=10)
        wordgrams_feats = word_vectorizer.fit_transform(sentences).astype('float32')


        # Store data in the hdf5 file
        f = h5py.File('multimodal_imdb.hdf5', mode='w')
        dtype = h5py.special_dtype(vlen=numpy.dtype('int32'))
        print("X.shape:",X.shape)
        print("Y.shape:",Y.shape)
        features = f.create_dataset('features', X.shape, dtype='float32')
        vgg_features = f.create_dataset(
            'vgg_features', (nsamples, 4096), dtype='float32')
        three_grams = f.create_dataset(
            'three_grams', (nsamples, ngrams_feats.shape[1]), dtype='float32')
        word_grams = f.create_dataset(
            'word_grams', (nsamples, wordgrams_feats.shape[1]), dtype='float32')
        images = f.create_dataset(
            'images', [nsamples, num_channels] + img_size[::-1], dtype='int32')
        seqs = f.create_dataset('sequences', (nsamples,), dtype=dtype)
        genres = f.create_dataset('genres', (nsamples, n_classes), dtype='int32')
        imdb_ids = f.create_dataset('imdb_ids', (nsamples,), dtype="S7")
        imdb_ids[...] = numpy.asarray([m['imdb_id']
                                    for m in movies], dtype='S7')[indices]
        features[...] = X
        for i, idx in enumerate(indices):
            images[i] = movies[idx]['cover']
            vgg_features[i] = movies[idx]['vgg_features']
        seqs[...] = sequences
        genres[...] = Y[indices][:, labels]
        self.dataset["genres"]= Y[indices][:, labels]
        three_grams[...] = ngrams_feats[indices].todense()
        word_grams[...] = wordgrams_feats[indices].todense()
        genres.attrs['target_names'] = json.dumps(target_names)
        features.dims[0].label = 'batch'
        features.dims[1].label = 'features'
        three_grams.dims[0].label = 'batch'
        three_grams.dims[1].label = 'features'
        word_grams.dims[0].label = 'batch'
        word_grams.dims[1].label = 'features'
        imdb_ids.dims[0].label = 'batch'
        genres.dims[0].label = 'batch'
        genres.dims[1].label = 'classes'
        vgg_features.dims[0].label = 'batch'
        vgg_features.dims[1].label = 'features'
        images.dims[0].label = 'batch'
        images.dims[1].label = 'channel'
        images.dims[2].label = 'height'
        images.dims[3].label = 'width'

        split_dict = {
            'train': {
                'features': (0, nsamples_train),
                'three_grams': (0, nsamples_train),
                'sequences': (0, nsamples_train),
                'images': (0, nsamples_train),
                'vgg_features': (0, nsamples_train),
                'imdb_ids': (0, nsamples_train),
                'word_grams': (0, nsamples_train),
                'genres': (0, nsamples_train)},
            'dev': {
                'features': (nsamples_train, nsamples_train + nsamples_dev),
                'three_grams': (nsamples_train, nsamples_train + nsamples_dev),
                'sequences': (nsamples_train, nsamples_train + nsamples_dev),
                'images': (nsamples_train, nsamples_train + nsamples_dev),
                'vgg_features': (nsamples_train, nsamples_train + nsamples_dev),
                'imdb_ids': (nsamples_train, nsamples_train + nsamples_dev),
                'word_grams': (nsamples_train, nsamples_train + nsamples_dev),
                'genres': (nsamples_train, nsamples_train + nsamples_dev)},
            'test': {
                'features': (nsamples_train + nsamples_dev, nsamples),
                'three_grams': (nsamples_train + nsamples_dev, nsamples),
                'sequences': (nsamples_train + nsamples_dev, nsamples),
                'images': (nsamples_train + nsamples_dev, nsamples),
                'vgg_features': (nsamples_train + nsamples_dev, nsamples),
                'imdb_ids': (nsamples_train + nsamples_dev, nsamples),
                'word_grams': (nsamples_train + nsamples_dev, nsamples),
                'genres': (nsamples_train + nsamples_dev, nsamples)}
        }

        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        f.flush()
        f.close()
        """


        ##
        #self.file = file
    
        self.start_ind = start_ind
        self.size = end_ind-start_ind
        self.vggfeature = vggfeature
        #self.dataset = h5py.File(self.file, 'r')

        """
        print("dataset type:",type(self.dataset))
        print(self.dataset["features"])
        print(self.dataset["images"])
        print(self.dataset["genres"])
        file_log_all = open("log_all.txt",'w',encoding='utf-8')
        file_log_all.write("self.dataset[features]:{}".format(self.dataset["features"][18160:25959]))
        file_log_all.write("self.dataset[images]:{}".format(self.dataset["images"][18160:25959]))
        file_log_all.write("self.dataset[genres]:{}".format(self.dataset["genres"][18160:25959]))
        
        file_log_all.close()
        """
    def __getitem__(self, ind):
        """Get item from dataset.

        Args:
            ind (int): Index of data to get

        Returns:
            tuple: Tuple of text input, image input, and label
        """
        #if not hasattr(self, 'dataset'):
            
        text = data_ori_dataset["features"][ind+self.start_ind]
        image = data_ori_dataset["images"][ind+self.start_ind] if not self.vggfeature else data_ori_dataset["vgg_features"][ind+self.start_ind]
            #
        label = data_ori_dataset["genres"][ind+self.start_ind]

        return text, image, label

    def __len__(self):
        """Get length of dataset."""
        return self.size


class IMDBDataset_robust(Dataset):
    """Implements a torch Dataset class for the imdb dataset that uses robustness measures as data augmentation."""

    def __init__(self, dataset, start_ind: int, end_ind: int) -> None:
        """Initialize IMDBDataset_robust object.

        Args:
            file (h5py.File): h5py file of data
            start_ind (int): Starting index for dataset
            end_ind (int): Ending index for dataset
            vggfeature (bool, optional): Whether to return pre-processed vgg_features or not. Defaults to False.
        """
        self.dataset = dataset
        self.start_ind = start_ind
        self.size = end_ind-start_ind

    def __getitem__(self, ind):
        """Get item from dataset.

        Args:
            ind (int): Index of data to get

        Returns:
            tuple: Tuple of text input, image input, and label
        """
        text = self.dataset[ind+self.start_ind][0]
        image = self.dataset[ind+self.start_ind][1]
        label = self.dataset[ind+self.start_ind][2]

        return text, image, label

    def __len__(self):
        """Get length of dataset."""
        return self.size


def _process_data(filename, path):
    data = {}
    filepath = os.path.join(path, filename)

    with Image.open(filepath+".jpeg") as f:
        image = np.array(f.convert("RGB"))
        data["image"] = image

    with open(filepath+".json", "r") as f:
        info = json.load(f)

        plot = info["plot"]
        data["plot"] = plot

    return data


def get_dataloader(device,path: str, test_path: str, num_workers: int = 8, train_shuffle: bool = True, batch_size: int = 40, vgg: bool = False, skip_process=False, no_robust=False) -> Tuple[Dict]:
    """Get dataloaders for IMDB dataset.

    Args:
        path (str): Path to training datafile.
        test_path (str): Path to test datafile.
        num_workers (int, optional): Number of workers to load data in. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        batch_size (int, optional): Batch size of data. Defaults to 40.
        vgg (bool, optional): Whether to return raw images or pre-processed vgg features. Defaults to False.
        skip_process (bool, optional): Whether to pre-process data or not. Defaults to False.
        no_robust (bool, optional): Whether to not use robustness measures as augmentation. Defaults to False.

    Returns:
        Tuple[Dict]: Tuple of Training dataloader, Validation dataloader, Test Dataloader
    """
    vgg=True
    """
    global data_ori

    
    
    with open("imdb_train_data", "wb") as fp:   #Pickling
        pickle.dump(data_ori, fp)
    with open("imdb_train_data", "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        print(b)
    """


    #data_ori=data_origin(device)
    global data_ori_dataset
    
    data_ori_dataset=h5py.File(path, 'r')
    print("shape:",data_ori_dataset["features"].shape)

    train_dataloader = DataLoader(IMDBDataset(device,path, 0, 15552, vgg),
                                  shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(IMDBDataset(device,path, 15552, 18160, vgg),
                                shuffle=False, num_workers=num_workers, batch_size=batch_size)
    if no_robust:
        test_dataloader = DataLoader(IMDBDataset(device,path, 18160, 25959, vgg),
                                     shuffle=False, num_workers=num_workers, batch_size=batch_size)
        return train_dataloader, val_dataloader, test_dataloader

    test_dataset = h5py.File(path, 'r')
    test_text = test_dataset['features'][18160:25959]
    test_vision = test_dataset['vgg_features'][18160:25959]
    labels = test_dataset["genres"][18160:25959]
    names = test_dataset["imdb_ids"][18160:25959]

    dataset = os.path.join(test_path, "dataset")

    if not skip_process:
        clsf = VGGClassifier(
            model_path='/home/pliang/multibench/MultiBench/datasets/imdb/vgg16.tar', synset_words='/home/xh/20220601_mmbench/MultiBench/datasets/imdb/synset_words.txt')
        googleword2vec = KeyedVectors.load_word2vec_format(
            '/home/xh/benchdata/Multimedia/mmimdb/GoogleNews-vectors-negative300.bin.gz', binary=True)

        images = []
        texts = []
        for name in tqdm(names):
            name = name.decode("utf-8")
            data = _process_data(name, dataset)
            images.append(data['image'])
            plot_id = np.array([len(p) for p in data['plot']]).argmax()
            texts.append(data['plot'][plot_id])

    # Add visual noises
    robust_vision = []
    for noise_level in range(11):
        vgg_filename = os.path.join(
            os.getcwd(), 'vgg_features_{}.npy'.format(noise_level))
        if not skip_process:
            vgg_features = []
            images_robust = add_visual_noise(
                images, noise_level=noise_level/10)
            for im in tqdm(images_robust):
                vgg_features.append(clsf.get_features(
                    Image.fromarray(im)).reshape((-1,)))
            np.save(vgg_filename, vgg_features)
        else:
            assert os.path.exists(vgg_filename) == True
            vgg_features = np.load(vgg_filename, allow_pickle=True)
        robust_vision.append([(test_text[i], vgg_features[i], labels[i])
                             for i in range(len(vgg_features))])

    test_dataloader = dict()
    test_dataloader['image'] = []
    for test in robust_vision:
        test_dataloader['image'].append(DataLoader(IMDBDataset_robust(test, 0, len(
            test)), shuffle=False, num_workers=num_workers, batch_size=batch_size))

    # Add text noises
    robust_text = []
    for noise_level in range(11):
        text_filename = os.path.join(
            os.getcwd(), 'text_features_{}.npy'.format(noise_level))
        if not skip_process:
            text_features = []
            texts_robust = add_text_noise(texts, noise_level=noise_level/10)
            for words in tqdm(texts_robust):
                words = words.split()
                if len([googleword2vec[w] for w in words if w in googleword2vec]) == 0:
                    text_features.append(np.zeros((300,)))
                else:
                    text_features.append(np.array(
                        [googleword2vec[w] for w in words if w in googleword2vec]).mean(axis=0))
            np.save(text_filename, text_features)
        else:
            assert os.path.exists(text_filename) == True
            text_features = np.load(text_filename, allow_pickle=True)
        robust_text.append([(text_features[i], test_vision[i], labels[i])
                           for i in range(len(text_features))])
    test_dataloader['text'] = []
    for test in robust_text:
        test_dataloader['text'].append(DataLoader(IMDBDataset_robust(test, 0, len(
            test)), shuffle=False, num_workers=num_workers, batch_size=batch_size))
    return train_dataloader, val_dataloader, test_dataloader
