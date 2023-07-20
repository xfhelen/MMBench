# Project:
#   VQA
# Description:
#   General purpose functions
# Author: 
#   Sergio Tascon-Morales

from tqdm import tqdm
import torch

def get_biomarker_name(data, question_id):
    for e in data:
        if e['question_id'] == question_id:
            biomarker = e['question_tokens'][2]
            break

    return biomarker


def group_answers_by_biomarker(answers, data):
    # function to get groups for input answers. Data is necessary to find the biomarker for a particular question id

    # first, analyze data to check how many different biomarkers are considered in the questions
    biomarkers = list(set([e['question_tokens'][2] for e in  data]))

    dict_groups = {b:[] for b in biomarkers}

    id_modans = answers['results']
    gtans_prob = answers['answers']
    for i in tqdm(range(id_modans.shape[0])): # for every answer
        q_id = id_modans[i, 0]
        biomarker = get_biomarker_name(data, q_id)
        dict_groups[biomarker].append(gtans_prob[i])

    return {k:torch.stack(v) for k, v in dict_groups.items()}


def get_question_type(data, question_id):
    for e in data:
        if e['question_id'] == question_id:
            t = e['question_type']
            gt_ans = e['answer']# assuming single answer
            break         
    return t, gt_ans

def group_answers_by_type(answers, data, map_answer_index):
    # divides answers into groups accroding to question types
    types = list(set(e['question_type'] for e in data))

    dict_groups_pred = {t:[] for t in types}
    dict_groups_gt = {t:[] for t in types}

    visited_ids = [] # to avoid repeated questions (which may appear when mainsub)
    for i in tqdm(range(answers.shape[0])):
        q_id = answers[i, 0].item()
        if q_id in visited_ids:
            continue
        typ, gt_ans = get_question_type(data, q_id)
        dict_groups_pred[typ].append(answers[i,1])
        dict_groups_gt[typ].append(torch.tensor(map_answer_index[gt_ans]))
        visited_ids.append(q_id)

    return {k:torch.stack(v) for k, v in dict_groups_pred.items()}, {k:torch.stack(v) for k, v in dict_groups_gt.items()}, types