# Project:
#   VQA
# Description:
#   QA processing for QA regions datasets (single, complementary and dual)
# Author: 
#   Sergio Tascon-Morales

from tqdm import tqdm
from collections import Counter
from . import nlp

def process_qa_regions_single(config, data_train, data_val, data_test):

    all_answers = [elem['answer'] for elem in data_train]

    # get top answers
    top_answers = nlp.get_top_answers(all_answers, config['num_answers'])

    # get maps for answers
    top_answers.append('UNK') # add unknown symbol answer
    map_answer_index = {elem:i for i, elem in enumerate(top_answers)}
    map_index_answer = top_answers.copy()

    # remove examples for which answer is not in top answers
    data_train = nlp.remove_examples_if_answer_not_common(data_train, top_answers) # should't remove anything because there are only two possible answers

    # tokenize questions for each subset
    print("Tokenizing questions...")
    data_train = nlp.add_tokens(data_train, config['tokenizer'])
    data_val = nlp.add_tokens(data_val, config['tokenizer'])
    data_test = nlp.add_tokens(data_test, config['tokenizer'])

    # insert UNK tokens and build word maps. No UNK tokens should be added since for this dataset the vocabulary is too limited
    print("Adding UNK tokens...")
    data_train, map_word_index, map_index_word = nlp.add_UNK_token_and_build_word_maps(data_train, config['min_word_frequency'])
    data_val = nlp.add_UNK_token(data_val, map_index_word)
    data_test = nlp.add_UNK_token(data_test, map_index_word)

    # encode questions
    print("Encoding questions...")
    data_train = nlp.encode_questions(data_train, map_word_index, config['max_question_length'])
    data_val = nlp.encode_questions(data_val, map_word_index, config['max_question_length'])
    data_test = nlp.encode_questions(data_test, map_word_index, config['max_question_length'])

    # encode answers
    print("Encoding answers...")
    data_train = nlp.encode_answers(data_train, map_answer_index)
    data_val = nlp.encode_answers(data_val, map_answer_index)

    # build return dictionaries
    sets = {'trainset': data_train, 'valset': data_val, 'testset': data_test}
    maps = {'map_index_word': map_index_word, 'map_word_index': map_word_index, 'map_index_answer': map_index_answer, 'map_answer_index': map_answer_index}

    # sets: {'trainset': trainset, 'valset': valset, 'testset': testset, 'testdevset': testdevset}
    # maps: {'map_index_word': map_index_word, 'map_word_index': map_word_index, 'map_index_answer': map_index_answer, 'map_answer_index': map_answer_index}
    return sets, maps 