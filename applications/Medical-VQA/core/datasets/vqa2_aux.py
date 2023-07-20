# Project:
#   VQA
# Description:
#   Auxiliary functions for dataset VQA2
# Author: 
#   Sergio Tascon-Morales

from tqdm import tqdm
from collections import Counter
from . import nlp
import itertools

def add_year(split='train'):
    if split in ['train', 'val']:
        return split + '2014'
    else:
        return 'test2015'        

def get_image_name(subtype='train2014', image_id='1', format='COCO_%s_%012d.jpg'):
    return format%(subtype, image_id)

def reformat_data(json_data, subset, annotations=[], mainsub=False):
    # function to reformat data (converts image id into image name, adds counting for answers)
    data = []
    print("Processing", subset, "data...")
    for i in tqdm(range(len(json_data))): # for every QA pair
        row = {}
        row['question_id'] = json_data[i]['question_id']
        row['image_name'] = get_image_name(add_year(subset), json_data[i]['image_id'])
        row['question'] = json_data[i]['question']
        if subset in ['train', 'val', 'trainval']:
            if mainsub:
                row['role'] = annotations[i]['role']
                if 'parent' in annotations[i]:
                    row['parent'] = annotations[i]['parent']
                else:
                    row['parent'] = None
            row['answer'] = annotations[i]['multiple_choice_answer']
            answers = []
            for ans in annotations[i]['answers']:
                answers.append(ans['answer'])
            row['answers_occurence'] = Counter(answers).most_common() # list of tuples with format [(AnsA, countA),(AnsB, countB)...(AnsN, countN)] where countA+countB+...+countN = 10
        data.append(row)
    return data

def reformat_data_mask(json_data, subset, annotations=[]):
    # function to reformat data (converts image id into image name, adds counting for answers) -> Special case with questions about regions
    data = []
    print("Processing", subset, "data...")
    for i in tqdm(range(len(json_data))): # for every QA pair
        row = {}
        row['question_id'] = json_data[i]['question_id']
        row['image_name'] = get_image_name(add_year(subset), json_data[i]['image_id'])
        row['question'] = json_data[i]['question']
        if subset in ['train', 'val', 'trainval']:
            row['answer'] = annotations[i]['multiple_choice_answer']
            answers = []
            for ans in annotations[i]['answers']:
                answers.append(ans['answer'])
            row['answers_occurence'] = Counter(answers).most_common() # list of tuples with format [(AnsA, countA),(AnsB, countB)...(AnsN, countN)] where countA+countB+...+countN = 10
        row['mask_name'] = json_data[i]['mask_name']
        data.append(row)

    return data


def process_qa_gqa(config, df):
    train_dict = df[df['subset'] == 'train'].to_dict('records')
    val_dict = df[df['subset'] == 'val'].to_dict('records')
    test_dict = df[df['subset'] == 'test'].to_dict('records')

    # get top answers
    all_answers = [e['ma'] for e in train_dict] + [e['sa'] for e in train_dict]
    top_answers = nlp.get_top_answers(all_answers, config['num_answers'])

    # get maps for answers
    top_answers.append('UNK') # add unknown symbol answer
    map_answer_index = {elem:i for i, elem in enumerate(top_answers)}
    map_index_answer = top_answers.copy()

    # tokenize questions for each pair
    print("Tokenizing questions...")
    for e in tqdm(train_dict):
        e['mq_tokens'] = nlp.tokenize_single_question(config['tokenizer'], e['mq'])
        e['sq_tokens'] = nlp.tokenize_single_question(config['tokenizer'], e['sq'])
    for e in tqdm(val_dict):
        e['mq_tokens'] = nlp.tokenize_single_question(config['tokenizer'], e['mq'])
        e['sq_tokens'] = nlp.tokenize_single_question(config['tokenizer'], e['sq'])
    for e in tqdm(test_dict):
        e['mq_tokens'] = nlp.tokenize_single_question(config['tokenizer'], e['mq'])
        e['sq_tokens'] = nlp.tokenize_single_question(config['tokenizer'], e['sq'])

    print('Creating maps and adding UNK token')
    all_words_in_all_questions = list(itertools.chain.from_iterable(elem['mq_tokens'] for elem in train_dict)) + list(itertools.chain.from_iterable(elem['sq_tokens'] for elem in train_dict))
    # count and sort 
    counts = Counter(all_words_in_all_questions).most_common()
    # get list of words (vocabulary for questions)
    vocab_words_in_questions = [elem[0] for elem in counts if elem[1] > config['min_word_frequency']]
    # add_entry for tokens with UNK to data
    for elem in tqdm(train_dict):
        elem['mq_tokens_with_UNK'] = [w if w in vocab_words_in_questions else 'UNK' for w in elem['mq_tokens']]
        elem['sq_tokens_with_UNK'] = [w if w in vocab_words_in_questions else 'UNK' for w in elem['sq_tokens']]
    # build maps
    vocab_words_in_questions.append('UNK') # Add UNK to the vocabulary
    map_word_index = {elem:i+1 for i,elem in enumerate(vocab_words_in_questions)} #* added +1 in i34 to avoid same symbol of padding
    map_index_word = {v:k for k,v in map_word_index.items()} 
    words_vocab_list = list(map_index_word.values())

    # now do the same for val and test
    for elem in tqdm(val_dict):
        elem['mq_tokens_with_UNK'] = [w if w in words_vocab_list else 'UNK' for w in elem['mq_tokens']]
        elem['sq_tokens_with_UNK'] = [w if w in words_vocab_list else 'UNK' for w in elem['sq_tokens']]
    for elem in tqdm(test_dict):
        elem['mq_tokens_with_UNK'] = [w if w in words_vocab_list else 'UNK' for w in elem['mq_tokens']]
        elem['sq_tokens_with_UNK'] = [w if w in words_vocab_list else 'UNK' for w in elem['sq_tokens']]

    # question Encoding
    print('Encoding questions')
    for elem in tqdm(train_dict):
        # add question length
        elem['mq_length'] = min(config['max_question_length'], len(elem['mq_tokens_with_UNK'])) # main
        elem['sq_length'] = min(config['max_question_length'], len(elem['sq_tokens_with_UNK'])) # sub
        elem['mq_word_indexes'] = [0]*config['max_question_length'] 
        elem['sq_word_indexes'] = [0]*config['max_question_length'] 
        for i, word in enumerate(elem['mq_tokens_with_UNK']):
            if i < config['max_question_length']:
                # using padding to the right. Add padding left?
                elem['mq_word_indexes'][i] = map_word_index[word] # replace word with index in vocabulary
        for i, word in enumerate(elem['sq_tokens_with_UNK']):
            if i < config['max_question_length']:
                # using padding to the right. Add padding left?
                elem['sq_word_indexes'][i] = map_word_index[word] # replace word with index in vocabulary
    for elem in tqdm(val_dict):
        # add question length
        elem['mq_length'] = min(config['max_question_length'], len(elem['mq_tokens_with_UNK'])) # main
        elem['sq_length'] = min(config['max_question_length'], len(elem['sq_tokens_with_UNK'])) # sub
        elem['mq_word_indexes'] = [0]*config['max_question_length'] 
        elem['sq_word_indexes'] = [0]*config['max_question_length'] 
        for i, word in enumerate(elem['mq_tokens_with_UNK']):
            if i < config['max_question_length']:
                # using padding to the right. Add padding left?
                elem['mq_word_indexes'][i] = map_word_index[word] # replace word with index in vocabulary
        for i, word in enumerate(elem['sq_tokens_with_UNK']):
            if i < config['max_question_length']:
                # using padding to the right. Add padding left?
                elem['sq_word_indexes'][i] = map_word_index[word] # replace word with index in vocabulary
    for elem in tqdm(test_dict):
        # add question length
        elem['mq_length'] = min(config['max_question_length'], len(elem['mq_tokens_with_UNK'])) # main
        elem['sq_length'] = min(config['max_question_length'], len(elem['sq_tokens_with_UNK'])) # sub
        elem['mq_word_indexes'] = [0]*config['max_question_length'] 
        elem['sq_word_indexes'] = [0]*config['max_question_length'] 
        for i, word in enumerate(elem['mq_tokens_with_UNK']):
            if i < config['max_question_length']:
                # using padding to the right. Add padding left?
                elem['mq_word_indexes'][i] = map_word_index[word] # replace word with index in vocabulary
        for i, word in enumerate(elem['sq_tokens_with_UNK']):
            if i < config['max_question_length']:
                # using padding to the right. Add padding left?
                elem['sq_word_indexes'][i] = map_word_index[word] # replace word with index in vocabulary

    print("Encoding answers...")
    for i, elem in enumerate(train_dict):
        unknown_answer_symbol = map_answer_index['UNK']
        elem['ma_index'] = map_answer_index.get(elem['ma'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers
        elem['sa_index'] = map_answer_index.get(elem['sa'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers
    for i, elem in enumerate(val_dict):
        unknown_answer_symbol = map_answer_index['UNK']
        elem['ma_index'] = map_answer_index.get(elem['ma'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers
        elem['sa_index'] = map_answer_index.get(elem['sa'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers
    for i, elem in enumerate(test_dict):
        unknown_answer_symbol = map_answer_index['UNK']
        elem['ma_index'] = map_answer_index.get(elem['ma'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers
        elem['sa_index'] = map_answer_index.get(elem['sa'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers

    sets = {'trainset': train_dict, 'valset': val_dict, 'testset': test_dict}
    maps = {'map_index_word': map_index_word, 'map_word_index': map_word_index, 'map_index_answer': map_index_answer, 'map_answer_index': map_answer_index}

    return sets, maps

def process_qa(config, data_train, data_val, data_test, data_testdev = None):
    # function to process questions and answers using functions from nlp.py This function can be used on other datasets
    all_answers = [elem['answer'] for elem in data_train]

    # get top answers
    top_answers = nlp.get_top_answers(all_answers, config['num_answers'])

    # get maps for answers
    top_answers.append('UNK') # add unknown symbol answer
    map_answer_index = {elem:i for i, elem in enumerate(top_answers)}
    map_index_answer = top_answers.copy()

    # remove examples for which answer is not in top answers
    # data_train = nlp.remove_examples_if_answer_not_common(data_train, top_answers)

    # tokenize questions for each subset
    print("Tokenizing questions...")
    data_train = nlp.add_tokens(data_train, config['tokenizer'])
    data_val = nlp.add_tokens(data_val, config['tokenizer'])
    data_test = nlp.add_tokens(data_test, config['tokenizer'])
    if data_testdev is not None:
        data_testdev = nlp.add_tokens(data_testdev, config['tokenizer'])

    # insert UNK tokens and build word maps
    print("Adding UNK tokens...")
    data_train, map_word_index, map_index_word = nlp.add_UNK_token_and_build_word_maps(data_train, config['min_word_frequency'])
    words_vocab_list = list(map_index_word.values())
    data_val = nlp.add_UNK_token(data_val, words_vocab_list)
    data_test = nlp.add_UNK_token(data_test, words_vocab_list)
    if data_testdev is not None:
        data_testdev = nlp.add_UNK_token(data_testdev, words_vocab_list)

    # encode questions
    print("Encoding questions...")
    data_train = nlp.encode_questions(data_train, map_word_index, config['max_question_length'])
    data_val = nlp.encode_questions(data_val, map_word_index, config['max_question_length'])
    data_test = nlp.encode_questions(data_test, map_word_index, config['max_question_length'])
    if data_testdev is not None:
        data_testdev = nlp.encode_questions(data_testdev, map_word_index, config['max_question_length'])

    # encode answers
    print("Encoding answers...")
    data_train = nlp.encode_answers(data_train, map_answer_index)
    data_val = nlp.encode_answers(data_val, map_answer_index)
    if 'answer' in data_test[0]: # if test set has answers
        data_test = nlp.encode_answers(data_test, map_answer_index)

    # build return dictionaries
    if data_testdev is not None:
        sets = {'trainset': data_train, 'valset': data_val, 'testset': data_test, 'testdevset': data_testdev}
    else:
        sets = {'trainset': data_train, 'valset': data_val, 'testset': data_test}
    maps = {'map_index_word': map_index_word, 'map_word_index': map_word_index, 'map_index_answer': map_index_answer, 'map_answer_index': map_answer_index}

    # sets: {'trainset': trainset, 'valset': valset, 'testset': testset, 'testdevset': testdevset}
    # maps: {'map_index_word': map_index_word, 'map_word_index': map_word_index, 'map_index_answer': map_index_answer, 'map_answer_index': map_answer_index}
    return sets, maps 