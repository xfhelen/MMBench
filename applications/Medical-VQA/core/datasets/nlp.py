# Project:
#   VQA
# Description:
#   NLP functions for processing questions and asnwers
# Author: 
#   Sergio Tascon-Morales

from collections import Counter 
import re
from tqdm import tqdm
from os.path import join as jp
import itertools
import json
import torch
import pickle
import os
from torch.utils.data import Dataset

def get_top_answers(answers, nans=2000):
    counts = Counter(answers).most_common()
    top_answers = [elem[0] for elem in counts[:nans]]
    return top_answers

def remove_examples_if_answer_not_common(qa_samples, top_answers):
    # removes qa_samples from qa_samples if their answer is not in top_answers
    after_removal = [elem for elem in qa_samples if elem['answer'] in top_answers]
    return after_removal

def clean_text(text):
    text = text.lower().replace("\n", " ").replace("\r", " ")
    # replace numbers and punctuation with space
    punc_list = '!"#$%&()*+,-./:;<=>?@[\]^_{|}~' + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.translate(t)

    # replace single quote with empty character
    t = str.maketrans(dict.fromkeys("'`", ""))
    text = text.translate(t)

    return text

def tokenizer_nltk(text, tokenizer):
    text = clean_text(text)
    tokens = tokenizer(text)
    return tokens

def tokenizer_spacy(text, tokenizer):
    text = clean_text(text)
    tokens = list(tokenizer(text))
    tokens_list_of_strings = [str(token) for token in tokens]
    return tokens_list_of_strings

def tokenizer_re(text):
    WORD = re.compile(r'\w+')
    text = clean_text(text)
    tokens = WORD.findall(text)
    return tokens

def tokenize_single_question(tokenizer_name, question_text):

    if tokenizer_name == 'nltk':
        from nltk import word_tokenize
    elif tokenizer_name == 'spacy':
        from spacy.tokenizer import Tokenizer
        from spacy.lang.en import English
        lang = English()
        tokenizer = Tokenizer(lang.vocab)

    if tokenizer_name == 'nltk':
        tokens = tokenizer_nltk(question_text, word_tokenize)
    elif tokenizer_name == 'spacy':
        tokens = tokenizer_spacy(question_text, tokenizer)
    elif tokenizer_name == 're':
        tokens = tokenizer_re(question_text)
    else:
        raise ValueError('Unknown tokenizer')
    return tokens

def add_tokens(qa_samples, tokenizer_name):
    if tokenizer_name == 'nltk':
        from nltk import word_tokenize
    elif tokenizer_name == 'spacy':
        from spacy.tokenizer import Tokenizer
        from spacy.lang.en import English
        lang = English()
        tokenizer = Tokenizer(lang.vocab)
    
    for elem in tqdm(qa_samples):
        question_text = elem['question']
        if tokenizer_name == 'nltk':
            elem['question_tokens'] = tokenizer_nltk(question_text, word_tokenize)
        elif tokenizer_name == 'spacy':
            elem['question_tokens'] = tokenizer_spacy(question_text, tokenizer)
        elif tokenizer_name == 're':
            elem['question_tokens'] = tokenizer_re(question_text)
        else:
            raise ValueError('Unknown tokenizer')
    
    return qa_samples

def add_UNK_token_and_build_word_maps(data, min_word_frequency):
    # function to build vocabulary from question words and then build maps to indexes
    all_words_in_all_questions = list(itertools.chain.from_iterable(elem['question_tokens'] for elem in data))
    # count and sort 
    counts = Counter(all_words_in_all_questions).most_common()
    # get list of words (vocabulary for questions)
    vocab_words_in_questions = [elem[0] for elem in counts if elem[1] > min_word_frequency]
    # add_entry for tokens with UNK to data
    for elem in tqdm(data):
        elem['question_tokens_with_UNK'] = [w if w in vocab_words_in_questions else 'UNK' for w in elem['question_tokens']]
    # build maps
    vocab_words_in_questions.append('UNK') # Add UNK to the vocabulary
    map_word_index = {elem:i+1 for i,elem in enumerate(vocab_words_in_questions)} #* added +1 in i34 to avoid same symbol of padding
    map_index_word = {v:k for k,v in map_word_index.items()} # * changed to dict in i34 because list does not work anymore due to first index being 1 instead of 0
    return data, map_word_index, map_index_word

def add_UNK_token_single(tokens, vocab):
    return [w if w in vocab else 'UNK' for w in tokens]

def add_UNK_token(data, vocab):
    for elem in tqdm(data):
        elem['question_tokens_with_UNK'] = [w if w in vocab else 'UNK' for w in elem['question_tokens']]
    return data

def encode_single_question(tokens, map_word_index, question_vector_length):
    question_length = min(question_vector_length, len(tokens))
    question_word_indexes = [0]*question_vector_length
    for i, word in enumerate(tokens):
        if i < question_vector_length:
            # using padding to the right. Add padding left?
            if word not in map_word_index:
                word = 'UNK'
            question_word_indexes[i] = map_word_index[word] # replace word with index in vocabulary
    return question_word_indexes

def encode_questions(data, map_word_index, question_vector_length):
    for elem in tqdm(data):
        # add question length
        elem['question_length'] = min(question_vector_length, len(elem['question_tokens_with_UNK']))
        elem['question_word_indexes'] = [0]*question_vector_length # create list with question_vector_length zeros
        for i, word in enumerate(elem['question_tokens_with_UNK']):
            if i < question_vector_length:
                # using padding to the right. Add padding left?
                elem['question_word_indexes'][i] = map_word_index[word] # replace word with index in vocabulary
    return data

def encode_answers(data, map_answer_index):
    # function to encode answers. If they are not in the answer vocab, they are mapped to -1
    if 'answers_occurence' in data[0]: # if there are multiple answers (VQA2 dataset)
        for i, elem in enumerate(data):
            answers = []
            answers_indexes = []
            answers_count = []
            unknown_answer_symbol = map_answer_index['UNK']
            elem['answer_index'] = map_answer_index.get(elem['answer'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers
            for answer in elem['answers_occurence']:
                answer_index = map_answer_index.get(answer[0], unknown_answer_symbol)
                #if answer_index != unknown_answer_symbol:
                answers += answer[1]*[answer[0]] # add all answers
                answers_indexes += answer[1]*[answer_index]
                answers_count.append(answer[1])
            elem['answers'] = answers 
            elem['answers_indexes'] = answers_indexes
            elem['answers_counts'] = answers_count
    else:
        for i, elem in enumerate(data):
            unknown_answer_symbol = map_answer_index['UNK']
            elem['answer_index'] = map_answer_index.get(elem['answer'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers
    return data



class NLPDataSet(Dataset):
    def __init__(self, config, subset):

        self.config = config
        self.subset = subset

        #paths
        self.path_data = config['path_qa']
        self.path_vocabs = config['path_vocabs']
        self.path_processed_output = jp(self.path_data, 'processed')
        if not os.path.exists(self.path_processed_output):
            os.mkdir(self.path_processed_output)

        self.data = {}

        # read raw data
        with open(jp(self.path_data, subset + '.json'), 'r') as f:
            self.data[subset] = json.load(f)

        # read vocabularies
        with open(jp(self.path_vocabs, 'map_word_index.pickle'), 'rb') as f:
            self.vocab_words = pickle.load(f)
        with open(jp(self.path_vocabs, 'map_answer_index.pickle'), 'rb') as f:
            self.vocab_answers = pickle.load(f)

        # encode questions if necessary and save files
        if not os.path.exists(jp(self.path_processed_output, subset + '.pickle')) or config['process_qa_again']:
            self.samples = self.process_and_save(self.data[self.subset], self.vocab_words, self.vocab_answers)
        else:
            # just read existing file
            with open(jp(self.path_processed_output, subset + '.pickle'), 'rb') as f:
                self.samples = pickle.load(f)


    def process_and_save(self, samples, vocab_words, vocab_answers):
        print('Processing samples for', self.subset, 'set...')
        # process train examples
        for sample in tqdm(samples):
            sample['mainq'] = encode_single_question(tokenize_single_question('re', sample['main_question']), vocab_words, self.config['max_question_length'])
            sample['maina'] = encode_single_question(tokenize_single_question('re', sample['main_answer']), vocab_words, 1)
            sample['subq'] = encode_single_question(tokenize_single_question('re', sample['sub_question']), vocab_words, self.config['max_question_length'])
            sample['suba'] = encode_single_question(tokenize_single_question('re', sample['sub_answer']), vocab_words, 1)            
        with open(jp(self.path_processed_output, self.subset + '.pickle'), 'wb') as f:
            pickle.dump(samples, f)        
        

        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        item = {  'main_question': torch.LongTensor(sample['mainq']),
                    'main_answer': torch.LongTensor(sample['maina']),
                    'sub_question': torch.LongTensor(sample['subq']),
                    'sub_answer': torch.LongTensor(sample['suba']),
                    'label': sample['label']}
        return item
    
    def __len__(self):
        return len(self.samples)
