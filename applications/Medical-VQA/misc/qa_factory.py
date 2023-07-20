# Project:
#   VQA
# Description:
#   QA generation functions
# Author: 
#   Sergio Tascon-Morales

import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from misc import dirs
from misc import image_processing as ip
from os.path import join as jp
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import shutil
from misc.image_processing import is_object_in_region, get_region_diameter
random.seed(1234)

DEFAULT_MASK_NAME = 'whole_image_mask.tif'

def map_coords(boxes, old_shape, new_shape):

    xmin, xmax, ymin, ymax = boxes
    Rx, Ry = new_shape[0]/old_shape[0], new_shape[1]/old_shape[1]
    new_xmin, new_xmax, new_ymin, new_ymax = round(xmin*Rx), round(xmax*Rx), round(ymin*Ry), round(ymax*Ry)
    new_boxes = [new_xmin, new_xmax, new_ymin, new_ymax]
    return new_boxes

def generate_random_window(w, h, min_w, max_w, min_h, max_h, prop, s, regions_in_subwindow=False, offset=0):
    p = random.random() # random number to decide whether random width or random height is sampled first
    if p >= 0.5:
        # sample width first
        random_w = random.randint(min_w, max_w)
        # to generate the random height I move back to the resized space so that the proportion is kept there instead of in the original space
        random_h = random.randint(max(round((s/h)*min_h), round((1-prop)*random_w*(s/w))), min(round(max_h*(s/h)), round((1+prop)*random_w*(s/w))))
        random_h = round((h/s)*random_h) # back to original space
    else:
        # sample height first
        random_h = random.randint(min_h, max_h)
        # to generate the random width I move back to the resized space so that the proportion is kept there instead of in the original space
        random_w = random.randint(max(round((s/w)*min_w), round((1-prop)*random_h*(s/h))), min(round(max_w*(s/w)), round((1+prop)*random_h*(s/h))))
        random_w = round((w/s)*random_w)

    if regions_in_subwindow:
        # force windows to be in a specific sub-range of the original image so that in the resized space they are also within a sub-range (a centered sub-window)
        top_left_corner = (random.randint(round((h/s)*offset), h - round((h/s)*offset) - random_h), random.randint(round((w/s)*offset), w - round((w/s)*offset) -random_w))
    else:
        top_left_corner = (random.randint(0, h - random_h), random.randint(0, w - random_w))
    return top_left_corner, random_w, random_h

def create_circular_mask(h, w, center=None, radius=None):
    """Function by alkasm on Stackoverflow: https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array"""
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def generate_circular_region_mask(image, od_diameter, whole_circle_within_image = True, chosen_center = None, chosen_radius = None):
    """Function to generate randomly located circular mask with random radius with value between 0.5*od_diameter and 1.5*od_diameter

    Parameters
    ----------
    image : numpy array
        fundus image
    od_diameter : float
        diameter of the optic disc for the current image
    whole_circle_within_image : boolean, default True
        whether or not the whole circular image should be contained in the image. If False, regions with area smaller than pi*r^2 may be generated specially at the bottom and top
    chosen_center : tuple 
        specific center for the region (if required). Default value is None

    Returns
    -------
    numpy array
        Binary mask with circular region of random radius and random location
    """

    if chosen_center is None and chosen_radius is None:
        # threshold image to get eye region (list of possible locations that are not background)
        mask_eye = image > 10 # threshold selected based on the intensities of the images

        # randomly choose radius for region in the range 0.5*od_diameter and 1.5*od_diameter
        radius_region = int(np.ceil((random.random() + 0.5)*od_diameter))

        # filter possible locations to exclude those that do not allow full circle to be on image
        if whole_circle_within_image:
            mask_eye[:radius_region, :] = False
            mask_eye[:, :radius_region] = False
            mask_eye[image.shape[0]-radius_region:, :] = False
            mask_eye[:, image.shape[1]-radius_region:] = False

        # list possible centers
        possible_centers = np.where(mask_eye)

        # randomly choose a center for the region
        center_index = random.randint(0, len(possible_centers[0]))
        center = (possible_centers[1][center_index], possible_centers[0][center_index]-1) # order is inverted to agree with function (next step)
    else: 
        radius_region = chosen_radius
        center = chosen_center

    # create binary mask based on center and radius
    mask = create_circular_mask(image.shape[0], image.shape[1], center = center, radius = radius_region)

    return mask


def get_question_from_class(class_name, template, suffix):
    if class_name in ['bench', 'bus', 'couch', 'sandwich', 'toothbrush', 'tooth brush', 'wineglass', 'wine glass']:
        plural = class_name + 'es'
    elif class_name in ['knife']:
        plural = 'knives'
    elif class_name in ['mouse']:
        plural = 'mice'
    elif class_name in ['scissors', 'skis']:
        plural = class_name
    else:
        plural = class_name + 's'

    if suffix == 'FC':
        question = 'is the fovea center in this region?'
    else:
        question = template.replace('<>', plural) + '?'
    
    return question


def generate_qa_single(config, subset, path_img, path_output_mask, img_index):
    # generation of questions based on the inside of a window for image in path_image according to information in config
    num_regions = config['num_regions']
    max_window_side_resized = config['max_window_side']
    min_window_side_resized = config['min_window_side']
    prop = config['proportion_deviation']
    resized_side = config['size']
    question_templates = config['question_templates']

    # read the image
    image = Image.open(path_img)
    h_orig = image.height
    w_orig = image.width

    # convert resized sizes to original dimensions
    max_window_width_orig = round((w_orig/resized_side)*max_window_side_resized)
    min_window_width_orig = round((w_orig/resized_side)*min_window_side_resized)
    max_window_height_orig = round((h_orig/resized_side)*max_window_side_resized)
    min_window_height_orig = round((h_orig/resized_side)*min_window_side_resized)

    cnt = num_regions*img_index # cnt gives offset for the ids of the masks

    qa_group = []  # list to collect all questions for current image

    for i_region in tqdm(range(num_regions)): # for each of the desired windows
        for i_q_type, (q_type, q_template) in enumerate(question_templates.items()):
            mask_id = cnt #? Why did I write this?
            top_left, window_w, window_h = generate_random_window(w_orig, h_orig, min_window_width_orig, max_window_width_orig, min_window_height_orig, max_window_height_orig, prop, resized_side)

            # build mask array
            mask_array = np.zeros((h_orig, w_orig), dtype = np.uint8)
            mask_array[top_left[0]:top_left[0]+window_h, top_left[1]:top_left[1]+window_w] = 1

            # define name for masks to be saved
            mask_id = "mask" + str(cnt).zfill(6) + ".tif"

            coords_norm = map_coords([top_left[1], top_left[1] + window_w, top_left[0], top_left[0]+window_h],
                                                            (w_orig, h_orig), (resized_side, resized_side)) # [xmin, xmax, ymin, ymax] x means horizontal axis (dim 1), ymin is vertical axis (dim 0)

            # generate resized masks and save them in corresponding folders
            mask_to_save = np.zeros((resized_side, resized_side), dtype=np.uint8)
            mask_to_save[coords_norm[2]:coords_norm[3], coords_norm[0]:coords_norm[1]] = 255
            Image.fromarray(mask_to_save).save(jp(path_output_mask, mask_id))

            # iterate through classes and get answers
            gt_available = True
            for i_class, (class_name, suffix) in enumerate(config['map_class_suffix'].items()): 
                question = q_template.replace('<>', class_name+'s') + '?'
                gt_location = jp(config['path_data'], 'masks',  subset, suffix, dirs.get_filename_without_extension(path_img) + '_' + suffix + '.' + config['format_gt'])
                if not os.path.exists(gt_location):
                    answer = 'no'
                    gt_available = False

                if gt_available:
                    gt_mask = Image.open(gt_location)
                    gt_mask_array = np.array(gt_mask)

                    if gt_mask_array.ndim > 2: # some gt images are RGBA, which makes necessary to take only the red dimension
                        gt_mask_array = gt_mask_array[:,:,0]  

                    num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array)
                    if num_pixels_in_region>0:
                        answer = 'yes'
                    else:
                        answer = 'no'

                qa_group.append({
                                    'image_name': dirs.get_filename_with_extension(path_img),
                                    'question': question,
                                    'question_id': int(str(i_class).zfill(2) + str(i_q_type).zfill(2) + str(i_region).zfill(4) + str(img_index).zfill(4)),
                                    'question_type': q_type,
                                    'mask_name': mask_id,
                                    'mask_coords': coords_norm,
                                    'answer': answer
                })
        cnt += 1

    return qa_group

def generate_qa_single_balanced(config, subset, path_img, path_output_mask, img_index, offset):
    # generation of questions based on the inside of a window for image in path_image according to information in config
    num_regions = config['num_regions']
    prop = config['proportion_deviation']
    resized_side = config['size']
    question_templates = config['question_templates']
    # if a threshold is specified, use it, otherwise use one pixel as threshold
    if 'threshold' in config:
        threshold = config['threshold'] #TODO: possible improvement, make threshold dependent on size of object (eg: 20% of pixels in object)
    else:
        threshold = 1

    # read the image
    image = Image.open(path_img)
    h_orig = image.height
    w_orig = image.width

    cnt = offset # cnt gives offset for the ids of the masks

    qa_group = []  # list to collect all questions for current image

    for i_q_type, (q_type, q_template) in enumerate(question_templates.items()): # for each question type (inside)
        for i_class, (class_name, suffix) in enumerate(config['map_class_suffix'].items()):  # for each abnormality
            print("Image", img_index+1, ", questions about", class_name)

            max_window_side_resized = config['max_window_side']
            min_window_side_resized = config['min_window_side']
            # convert resized sizes to original dimensions
            max_window_width_orig = round((w_orig/resized_side)*max_window_side_resized)
            min_window_width_orig = round((w_orig/resized_side)*min_window_side_resized)
            max_window_height_orig = round((h_orig/resized_side)*max_window_side_resized)
            min_window_height_orig = round((h_orig/resized_side)*min_window_side_resized)

            num_questions_yes = 0
            num_questions_no = 0
            # check if class is present in the current image
            gt_location = jp(config['path_data'], 'masks',  subset, suffix, dirs.get_filename_without_extension(path_img) + '_' + suffix + '.' + config['format_gt'])
            if not os.path.exists(gt_location):
                print("No GT, skipping for this category...")
                continue # continue if gt does not exist for this patients because all questions would be answered with NO, contributing to the imbalance
            # if gt exists, start generating positive questions
            cnt_since_last_yes = 0
            cnt_since_last_no = 0
            i_region = 0
            window_size_changes = 0 # number of times that the window has changed it's size
            if 'max_window_changes' in config:
                max_window_size_changes = config['max_window_changes']
            else:
                max_window_size_changes = 10 # default maximum of times that the window size can be modified when it get's to hard to find a yes-question or a no-question
            while num_questions_yes < round(num_regions/2) or num_questions_no < round(num_regions/2):
                top_left, window_w, window_h = generate_random_window(w_orig, h_orig, min_window_width_orig, max_window_width_orig, min_window_height_orig, max_window_height_orig, prop, resized_side, regions_in_subwindow=True, offset=config['window_offset'])
                # build mask array
                mask_array = np.zeros((h_orig, w_orig), dtype = np.uint8)
                mask_array[top_left[0]:top_left[0]+window_h, top_left[1]:top_left[1]+window_w] = 1
                gt_mask = Image.open(gt_location)
                gt_mask_array = np.array(gt_mask)

                # if GT is almost completely white (object occupies almost whole picture), then just skip
                if np.count_nonzero(gt_mask_array) > 0.9*h_orig*w_orig:
                    print("GT too big, too hard or impossible to find NO-questions. Skipping category")
                    break

                if gt_mask_array.ndim > 2: # some gt images are RGBA in iDRID, which makes necessary to take only the red dimension
                    gt_mask_array = gt_mask_array[:,:,0]  
                
                # generate answer
                num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array)
                #if 'idrid' in path_img.lower():
                #    object_in_region = False
                #else:
                #    object_in_region = is_object_in_region(gt_mask_array, mask_array) #! bottleneck

                if (num_pixels_in_region >= threshold) and num_questions_yes < round(num_regions/2): # if answer is yes and i haven't reached the maximum number of positive questions
                    answer = 'yes'
                    mask_id = "mask" + str(cnt).zfill(6) + ".tif"

                    # build plurals for text questions
                    if class_name in ['bench', 'bus', 'couch', 'sandwich', 'toothbrush', 'tooth brush', 'wineglass', 'wine glass']:
                        plural = class_name + 'es'
                    elif class_name in ['knife']:
                        plural = 'knives'
                    elif class_name in ['mouse']:
                        plural = 'mice'
                    elif class_name in ['scissors', 'skis']:
                        plural = class_name
                    else:
                        plural = class_name + 's'

                    if suffix == 'FC':
                        question = 'is the fovea center in this region?'
                    else:
                        question = q_template.replace('<>', plural) + '?'

                    coords_norm = map_coords([top_left[1], top_left[1] + window_w, top_left[0], top_left[0]+window_h],
                                                        (w_orig, h_orig), (resized_side, resized_side)) # [xmin, xmax, ymin, ymax] x means horizontal axis (dim 1), ymin is vertical axis (dim 0)

                    # generate resized masks and save them in corresponding folders
                    mask_to_save = np.zeros((resized_side, resized_side), dtype=np.uint8)
                    mask_to_save[coords_norm[2]:coords_norm[3], coords_norm[0]:coords_norm[1]] = 255
                    Image.fromarray(mask_to_save).save(jp(path_output_mask, mask_id))

                    qa_group.append({
                                        'image_name': dirs.get_filename_with_extension(path_img),
                                        'question': question,
                                        'question_id': int(str(i_class).zfill(2) + str(i_q_type).zfill(2) + str(i_region).zfill(4) + str(img_index).zfill(4)),
                                        'question_type': q_type,
                                        'mask_name': mask_id,
                                        'mask_coords': coords_norm,
                                        'answer': answer
                    })
                    num_questions_yes += 1
                    cnt_since_last_yes = 0
                    cnt_since_last_no += 1
                    cnt += 1
                    i_region += 1

                elif num_pixels_in_region == 0 and num_questions_no < round(num_regions/2): # if answer is no and I haven't reached the max. number of negative questions
                    answer = 'no'
                    mask_id = "mask" + str(cnt).zfill(6) + ".tif"

                    if suffix == 'FC':
                        question = 'is the fovea center in this region?'
                    else:
                        question = q_template.replace('<>', class_name+'s') + '?'

                    coords_norm = map_coords([top_left[1], top_left[1] + window_w, top_left[0], top_left[0]+window_h],
                                                        (w_orig, h_orig), (resized_side, resized_side)) # [xmin, xmax, ymin, ymax] x means horizontal axis (dim 1), ymin is vertical axis (dim 0)

                    # generate resized masks and save them in corresponding folders
                    mask_to_save = np.zeros((resized_side, resized_side), dtype=np.uint8)
                    mask_to_save[coords_norm[2]:coords_norm[3], coords_norm[0]:coords_norm[1]] = 255
                    Image.fromarray(mask_to_save).save(jp(path_output_mask, mask_id))

                    qa_group.append({
                                        'image_name': dirs.get_filename_with_extension(path_img),
                                        'question': question,
                                        'question_id': int(str(i_class).zfill(2) + str(i_q_type).zfill(2) + str(i_region).zfill(4) + str(img_index).zfill(4)),
                                        'question_type': q_type,
                                        'mask_name': mask_id,
                                        'mask_coords': coords_norm,
                                        'answer': answer
                    })
                    num_questions_no += 1
                    cnt_since_last_no = 0
                    cnt_since_last_yes += 1
                    cnt += 1
                    i_region += 1
                else:
                    cnt_since_last_no += 1
                    cnt_since_last_yes += 1

                    if cnt_since_last_no >= 20*num_regions and num_questions_no < round(num_regions/2): # if after looking for a "no" answer it didn't come, make window 10% smaller
                        if window_size_changes < max_window_size_changes:
                            print("\t After", 20*num_regions, "attempts for finding a negative question, reducing window size to min:", round(min_window_side_resized*0.9), ', max:', round(max_window_side_resized*0.9))
                            max_window_side_resized = round(max_window_side_resized*0.9)
                            min_window_side_resized = round(min_window_side_resized*0.9)
                            assert min_window_side_resized < max_window_side_resized

                            max_window_width_orig = round((w_orig/resized_side)*max_window_side_resized)
                            min_window_width_orig = round((w_orig/resized_side)*min_window_side_resized)
                            max_window_height_orig = round((h_orig/resized_side)*max_window_side_resized)
                            min_window_height_orig = round((h_orig/resized_side)*min_window_side_resized)
                            cnt_since_last_no = 0
                            window_size_changes += 1
                        else:
                            print("\t Maximum number of changes in window reached")
                            if cnt_since_last_no > 50*num_regions:
                                print("\t Jumping to next category")
                                break
                    if cnt_since_last_yes >= 20*num_regions and num_questions_yes < round(num_regions/2):
                        if window_size_changes < max_window_size_changes:
                            print("\t After", 20*num_regions, "attempts for finding a positive question, increasing window size to min:", round(min_window_side_resized*1.1), ', max:', round(max_window_side_resized*1.1))
                            max_window_side_resized = round(max_window_side_resized*1.1)
                            min_window_side_resized = round(min_window_side_resized*1.1)
                            assert min_window_side_resized < max_window_side_resized

                            max_window_width_orig = round((w_orig/resized_side)*max_window_side_resized)
                            min_window_width_orig = round((w_orig/resized_side)*min_window_side_resized)
                            max_window_height_orig = round((h_orig/resized_side)*max_window_side_resized)
                            min_window_height_orig = round((h_orig/resized_side)*min_window_side_resized)
                            cnt_since_last_yes = 0
                            window_size_changes += 1
                        else:
                            print("\t Maximum number of changes in window reached")
                            if cnt_since_last_yes > 50*num_regions:
                                print("\t Jumping to next category")
                                break
    return qa_group


def generate_dme_qa_single(config, subset, path_img, path_output_mask, img_index):
    # generate qa pairs without balancing them (for test) both about random regions and about whole image
    num_regions = config['num_regions']
    resized_side = config['size']
    question_templates = config['question_templates']
    cnt = num_regions*img_index # cnt gives offset for the ids of the masks
    image = Image.open(path_img)
    qa_group = []  # list to collect all questions for current image

    if 'threshold' in config:
        threshold = config['threshold'] #TODO: possible improvement, make threshold dependent on size of object (eg: 20% of pixels in object)
    else:
        threshold = 1

    # create default full size mask for questions about whole image and questions about DME grade
    if img_index == 0:
        whole_img_mask = Image.fromarray(255*np.ones((resized_side, resized_side)).astype(np.uint8))
        whole_img_mask.save(jp(path_output_mask, 'whole_image_mask.tif'), 'TIFF')

    od_gt = jp(config['path_data'], 'masks',  subset, 'OD', dirs.get_filename_without_extension(path_img) + '_' + 'OD' + '.' + config['format_gt'])
    od_img = Image.open(od_gt)
    od_img_np = np.array(od_img).astype(np.bool)

    # get OD diameter in original space
    od_diameter, _, _ = get_region_diameter(od_img_np)

    for i_region in tqdm(range(num_regions)): # for each of the desired windows
        for i_q_type, (q_type, q_template) in enumerate(question_templates.items()):    

            # generate circular region
            mask_array = generate_circular_region_mask(np.array(ImageOps.grayscale(image)), od_diameter)

            # define name for masks to be saved
            mask_id = "mask" + str(cnt).zfill(6) + ".tif"

            # save mask
            mask_to_save = cv2.resize(mask_array.astype(np.uint8), dsize=(resized_side, resized_side), interpolation=cv2.INTER_NEAREST)*255
            Image.fromarray(mask_to_save).save(jp(path_output_mask, mask_id))

            gt_available = True
            for i_class, (class_name, suffix) in enumerate(config['map_class_suffix'].items()): 
                gt_location = jp(config['path_data'], 'masks',  subset, suffix, dirs.get_filename_without_extension(path_img) + '_' + suffix + '.' + config['format_gt'])
                if not os.path.exists(gt_location):
                    answer = 'no'
                    gt_available = False

                if gt_available:
                    gt_mask = Image.open(gt_location)
                    gt_mask_array = np.array(gt_mask)

                    if gt_mask_array.ndim > 2: # some gt images are RGBA, which makes necessary to take only the red dimension
                        gt_mask_array = gt_mask_array[:,:,0]  

                    # here I can generate the question about the whole image
                    if suffix != 'OD' and i_region==0:
                        answer = 'yes'
                        if np.count_nonzero(gt_mask_array) < threshold:
                            answer = 'no'

                        # get question ensuring template is the right for the requierd question
                        question = get_question_from_class(class_name, 'are there <> in this image', suffix)

                        qa_group.append({
                                            'image_name': dirs.get_filename_with_extension(path_img),
                                            'question': question,
                                            'question_id': int(str(i_class).zfill(2) + str(11).zfill(2) + str(i_region).zfill(4) + str(img_index).zfill(4)),
                                            'question_type': 'whole', # fix q type
                                            'mask_name': "whole_image_mask.tif", # fix mask name
                                            'answer': answer
                        })
                        i_region += 1

                    # now question about region
                    num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array)
                    if num_pixels_in_region>threshold:
                        answer = 'yes'
                    else:
                        answer = 'no'
                
                question =  get_question_from_class(class_name, q_template, suffix)

                qa_group.append({
                                    'image_name': dirs.get_filename_with_extension(path_img),
                                    'question': question,
                                    'question_id': int(str(i_class).zfill(2) + str(i_q_type).zfill(2) + str(i_region).zfill(4) + str(img_index).zfill(4)),
                                    'question_type': q_type,
                                    'mask_name': mask_id,
                                    'answer': answer
                })                    
        cnt += 1                  
    return qa_group

def generate_question_id(i_class, i_q_type, i_region, img_index):
    # pre-defined question id generated from class index, question type index, region index and image index
    return int(str(i_class).zfill(2) + str(i_q_type).zfill(2) + str(i_region).zfill(4) + str(img_index).zfill(4))

def generate_dict_qa(image_name_w_ext, question, question_id, question_type, mask_id, answer, center='none', role='ind'):
    di =  {
                                        'image_name': image_name_w_ext,
                                        'question': question,
                                        'question_id': question_id,
                                        'question_type': question_type,
                                        'mask_name': mask_id,
                                        'answer': answer,
                                        'center': center,
                                        'role': role # Adding role field (i57)
                    }
    return di


def resize_and_save(mask_array, resized_side, path_output):                    
    # resize to square size, multiply by 255 (required values 0,1) 
    mask_to_save = cv2.resize(mask_array.astype(np.uint8), dsize=(resized_side, resized_side), interpolation=cv2.INTER_NEAREST)*255
    Image.fromarray(mask_to_save).save(path_output)


def generate_dme_qa_single_inside_balanced(config, subset, path_img_nh, path_img_h, path_output_mask, img_index, offset, fovea_centers, dme_grades):
    # generation of questions based on the inside of a window for image in path_image according to information in config
    num_regions = config['num_regions']
    resized_side = config['size']
    question_templates = config['question_templates']
    # if a threshold is specified, use it, otherwise use one pixel as threshold
    if 'threshold' in config:
        threshold = config['threshold'] #TODO: possible improvement, make threshold dependent on size of object (eg: 20% of pixels in object)
    else:
        threshold = 1

    available_fovea_centers = list(fovea_centers['image_id'])

    # read the unhealthy image
    image_nh = Image.open(path_img_nh)
    h_orig = image_nh.height
    w_orig = image_nh.width

    # prepare list of healthy images
    image_h = Image.open(path_img_h)
    healthy_image_name_wo_ext = dirs.get_filename_without_extension(path_img_h)
    unhealthy_image_name_wo_ext = dirs.get_filename_without_extension(path_img_nh)

    # create default full size mask for questions about whole image and questions about DME grade
    if img_index == 0:
        whole_img_mask = Image.fromarray(255*np.ones((resized_side, resized_side)).astype(np.uint8))
        whole_img_mask.save(jp(path_output_mask, 'whole_image_mask.tif'), 'TIFF')

    cnt = offset # cnt gives offset for the ids of the masks

    qa_group = []  # list to collect all questions for current image

    for i_q_type, (q_type, q_template) in enumerate(question_templates.items()): # for each question type (inside)
        for i_class, (class_name, suffix) in enumerate(config['map_class_suffix'].items()):  # for each abnormality
            print("Image", img_index+1, ", questions about", class_name)

            # Since I am using circular regions, and since not all images have the same size, it seems convenient to define the maximum radius 
            # for the regions based on the radius of the optic disc of every image. This implies I have to have a function to measure the radius of the OD
            # from the GT

            # read OD mask from masks folder 
            od_gt = jp(config['path_data'], 'masks',  subset, 'OD', unhealthy_image_name_wo_ext + '_' + 'OD' + '.' + config['format_gt'])
            od_img = Image.open(od_gt)
            od_img_np = np.array(od_img).astype(np.bool)

            # get OD diameter in original space
            od_diameter, _, _ = get_region_diameter(od_img_np)

            num_questions_yes = 0
            num_questions_no = 0

            # check if class is present in current image
            gt_location = jp(config['path_data'], 'masks',  subset, suffix, unhealthy_image_name_wo_ext + '_' + suffix + '.' + config['format_gt'])
            if not os.path.exists(gt_location):
                print("No GT, skipping for this category...")
                continue # continue if gt does not exist for this patients because all questions would be answered with NO, contributing to the imbalance

            i_region = 0

            # get question
            question =  get_question_from_class(class_name, q_template, suffix)

            if suffix == 'OD':
                patience = 0
            else:
                patience = np.inf # so that only for OD questions the patience is used (i36)

            while num_questions_yes < round(num_regions/2) or patience < config['patience_multiplier']*num_regions: #* Big change: This while now considers only amount of Yes-answer questions because the balance will be performed afterwards.
                                                                                         #* and added patience because in some cases for OD result is not balanced (which should be) 

                mask_array = generate_circular_region_mask(np.array(ImageOps.grayscale(image_nh)), od_diameter)
                gt_mask = Image.open(gt_location)
                gt_mask_array = np.array(gt_mask)

                # if GT is almost completely white (object occupies almost whole picture), then just skip
                if np.count_nonzero(gt_mask_array) > 0.9*h_orig*w_orig:
                    print("GT too big, too hard or impossible to find NO-questions. Skipping category")
                    break

                if gt_mask_array.ndim > 2: # some gt images are RGBA in iDRID, which makes necessary to take only the red dimension
                    gt_mask_array = gt_mask_array[:,:,0]  

                # generate answer
                num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array)

                if num_pixels_in_region >= threshold and num_questions_yes < round(num_regions/2): # if answer is yes and i haven't reached the maximum number of positive questions
                    answer = 'yes'
                    mask_id = "mask" + str(cnt).zfill(6) + ".tif"

                    # generate resized masks and save them in corresponding folders
                    resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                    q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
                    qa_group.append(generate_dict_qa(unhealthy_image_name_wo_ext + '.jpg', question, q_id, q_type, mask_id, answer, center='random'))

                    num_questions_yes += 1
                    cnt += 1
                    i_region += 1
                # consider also negative questions, just in case they appear (eg from the borders of the eye, probably)
                elif num_pixels_in_region == 0 and num_questions_no < round(num_regions/2): # if answer is no and I haven't reached the max. number of negative questions
                    answer = 'no'
                    mask_id = "mask" + str(cnt).zfill(6) + ".tif"               

                    # generate resized masks and save them in corresponding folders
                    resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                    q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
                    qa_group.append(generate_dict_qa(unhealthy_image_name_wo_ext + '.jpg', question, q_id, q_type, mask_id, answer, center='random'))
                    num_questions_no += 1
                    cnt += 1
                    i_region += 1
                    patience += 1
                
                # if balance is reached, stop trying
                if num_questions_yes == round(num_regions/2) and num_questions_no == round(num_regions/2):
                    break

            # here while loop finishes, I should check how many YES and NO questions were appended (check num_questions_i) and then compensate using healthy images.
            # At this point I know that the number of YES-questions is N/2. Number of NO-questions is anything between 0 and N/2.
            if num_questions_no < round(num_regions/2) and suffix != 'OD': # if not balanced, add questions
                to_add = round(num_regions/2) - num_questions_no

                # now, I have to use healthy images to balance the dataset. This is better than searching a lot in the same unhealthy image. 
                for i_bal in range(to_add):

                    # do same process with healthy image to generate no-answers 
                    od_gt = jp(config['path_data'], 'masks',  subset, 'OD', healthy_image_name_wo_ext + '_' + 'OD' + '.' + config['format_gt'])
                    od_img = Image.open(od_gt)
                    od_img_np = np.array(od_img).astype(np.bool)

                    # get OD diameter in original space
                    od_diameter, _, _ = get_region_diameter(od_img_np)

                    # generate mask
                    mask_array = generate_circular_region_mask(np.array(ImageOps.grayscale(image_h)), od_diameter)

                    # In this case I know that the biomarker is not OD and I know that the image is healthy, therefore the answer has to be No
                    question =  get_question_from_class(class_name, q_template, suffix)

                    # save answer to qa_group
                    answer = 'no'
                    mask_id = "mask" + str(cnt).zfill(6) + ".tif"    
                    role = 'sub' # in this case I know that the image is healthy, meaning grade is 0, meaning all inside questions are sub-questions            

                    # generate resized masks and save them in corresponding folders
                    resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                    q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
                    qa_group.append(generate_dict_qa(healthy_image_name_wo_ext + '.jpg', question, q_id, q_type, mask_id, answer, center='random', role=role))
                    num_questions_no += 1
                    cnt += 1
                    i_region += 1

            if suffix == 'OD' and num_questions_no != num_questions_yes: # sanity check
                raise Exception("What should not happen happened :O")

    # Now add two questions about regions centered at macula if annotation about macula is available
    # first for unhealthy image
    suffix = 'EX'
    q_type = 'inside'
    question = 'are there hard exudates in this region?'
    if unhealthy_image_name_wo_ext in available_fovea_centers: 
        macula_loc = (int(fovea_centers[fovea_centers['image_id'] == unhealthy_image_name_wo_ext]['x']), int(fovea_centers[fovea_centers['image_id'] == unhealthy_image_name_wo_ext]['y']))
        # if I have the fovea center, generate two questions with center at macula and radius smaller than assumed OD diameter.
        for r in [0.5*od_diameter, 0.9*od_diameter]:   
            mask_array = generate_circular_region_mask(np.array(ImageOps.grayscale(image_nh)), od_diameter, chosen_center=macula_loc, chosen_radius=r)         
            gt_location = jp(config['path_data'], 'masks',  subset, suffix, unhealthy_image_name_wo_ext + '_' + suffix + '.' + config['format_gt'])
            gt_mask = Image.open(gt_location)
            gt_mask_array = np.array(gt_mask)
            num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array)

            if num_pixels_in_region >= threshold:
                answer = 'yes'
                mask_id = "mask" + str(cnt).zfill(6) + ".tif"

                role = 'sub'

                # generate resized masks and save them in corresponding folders
                resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
                qa_group.append(generate_dict_qa(unhealthy_image_name_wo_ext + '.jpg', question, q_id, q_type, mask_id, answer, center='macula', role=role))

                cnt += 1
                i_region += 1       

            elif num_pixels_in_region == 0:   
                answer = 'no'
                mask_id = "mask" + str(cnt).zfill(6) + ".tif"               

                role = 'sub'

                # generate resized masks and save them in corresponding folders
                resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
                qa_group.append(generate_dict_qa(unhealthy_image_name_wo_ext + '.jpg', question, q_id, q_type, mask_id, answer, center='macula', role=role))
                cnt += 1
                i_region += 1


    # now for healthy image
    if healthy_image_name_wo_ext in available_fovea_centers:
        macula_loc = (int(fovea_centers[fovea_centers['image_id'] == healthy_image_name_wo_ext]['x']), int(fovea_centers[fovea_centers['image_id'] == healthy_image_name_wo_ext]['y']))
        for r in [0.5*od_diameter, 0.9*od_diameter]:   
            mask_array = generate_circular_region_mask(np.array(ImageOps.grayscale(image_h)), od_diameter, chosen_center=macula_loc, chosen_radius=r)         
            # in this case I don't have gt, but know that because image is healthy, answer must be 0
            answer = 'no'
            mask_id = "mask" + str(cnt).zfill(6) + ".tif"               
            role = 'sub' # healthy image, hence grade is zero, hence all inside questions, even if centered at macula, are sub-questions
            # generate resized masks and save them in corresponding folders
            resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

            q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
            qa_group.append(generate_dict_qa(healthy_image_name_wo_ext + '.jpg', question, q_id, q_type, mask_id, answer, center='macula', role=role))
            cnt += 1
            i_region += 1

    return qa_group

def generate_dme_qa_single_known_answer(config, subset, image_name_wo_ext, path_image, path_output_mask, test_image_index, offset, fovea_centers, answer='no'):
    # function to create QA pairs when the answer is known. Create N qa pairs about random regions, then some for regions centered at macula (with reasonable radius)
    # generation of questions based on the inside of a window for image in path_image according to information in config
    num_regions = config['num_regions']
    resized_side = config['size']
    question_templates = config['question_templates']
    # if a threshold is specified, use it, otherwise use one pixel as threshold
    if 'threshold' in config:
        threshold = config['threshold'] #TODO: possible improvement, make threshold dependent on size of object (eg: 20% of pixels in object)
    else:
        threshold = 1

    image_name_w_ext = image_name_wo_ext + '.jpg'

    available_fovea_centers = list(fovea_centers['image_id'])

    # create test whole image mask (this part was added after generating version 3 of dataset)
    if test_image_index == 0:
        whole_img_mask = Image.fromarray(255*np.ones((resized_side, resized_side)).astype(np.uint8))
        whole_img_mask.save(jp(path_output_mask, 'whole_image_mask.tif'), 'TIFF')

    # read the unhealthy image
    image_nh = Image.open(path_image)
    h_orig = image_nh.height
    w_orig = image_nh.width 

    cnt = offset # cnt gives offset for the ids of the masks

    qa_group = []  # list to collect all questions for current image   

    for i_q_type, (q_type, q_template) in enumerate(question_templates.items()): # for each question type (inside)
        for i_class, (class_name, suffix) in enumerate({'hard exudate': 'EX'}.items()):  # Only for EX
            print("Image", test_image_index+1, ", questions about", class_name)

            # read OD mask from masks folder 
            od_gt = jp(config['path_data'], 'masks',  subset, 'OD', dirs.get_filename_without_extension(image_name_wo_ext) + '_' + 'OD' + '.' + config['format_gt'])
            if not os.path.exists(od_gt):
                raise FileNotFoundError
            od_img = Image.open(od_gt)
            od_img_np = np.array(od_img).astype(np.bool)

            # get OD diameter in original space
            od_diameter, _, _ = get_region_diameter(od_img_np)

            normal_question_counter = 0

            i_region = 0

            # get question
            question =  get_question_from_class(class_name, q_template, suffix)

            while normal_question_counter < round(num_regions/2):
                mask_array = generate_circular_region_mask(np.array(ImageOps.grayscale(image_nh)), od_diameter)

                # In this case I do not need to find the answer

                mask_id = "mask" + str(cnt).zfill(6) + ".tif"               
                role = 'sub'
                # generate resized masks and save them in corresponding folders
                resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                q_id = generate_question_id(i_class, i_q_type, i_region, test_image_index)
                qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, center='random', role=role))
                normal_question_counter += 1
                cnt += 1
                i_region += 1

            if image_name_wo_ext in available_fovea_centers:
                macula_loc = (int(fovea_centers[fovea_centers['image_id'] == image_name_wo_ext]['x']), int(fovea_centers[fovea_centers['image_id'] == image_name_wo_ext]['y']))
                # if I have the fovea center, generate two questions with center at macula and radius smaller than assumed OD diameter.
                for r in [0.5*od_diameter, 0.9*od_diameter]:
                    mask_array = generate_circular_region_mask(np.array(ImageOps.grayscale(image_nh)), od_diameter, chosen_radius=r, chosen_center=macula_loc)

                    mask_id = "mask" + str(cnt).zfill(6) + ".tif"   
                    role = 'sub'

                    # generate resized masks and save them in corresponding folders
                    resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                    q_id = generate_question_id(i_class, i_q_type, i_region, test_image_index)
                    qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, center='macula', role=role))
                    cnt += 1
                    i_region += 1

    return qa_group

def generate_dme_qa_single_test(config, subset, image_name_wo_ext, path_image, path_output_mask, test_image_index, offset, fovea_centers):
    # function to create QA pairs about regions centered at random locations, and also some centered at macula, with radius < 1 OD diameter
    num_regions = config['num_regions']
    resized_side = config['size']
    question_templates = config['question_templates']
    # if a threshold is specified, use it, otherwise use one pixel as threshold
    if 'threshold' in config:
        threshold = config['threshold'] #TODO: possible improvement, make threshold dependent on size of object (eg: 20% of pixels in object)
    else:
        threshold = 1

    image_name_w_ext = image_name_wo_ext + '.jpg'

    available_fovea_centers = list(fovea_centers['image_id'])

    # create test whole image mask (this part was added after generating version 3 of dataset)
    if test_image_index == 0:
        whole_img_mask = Image.fromarray(255*np.ones((resized_side, resized_side)).astype(np.uint8))
        whole_img_mask.save(jp(path_output_mask, 'whole_image_mask.tif'), 'TIFF')

    # read the unhealthy image
    image_nh = Image.open(path_image)
    h_orig = image_nh.height
    w_orig = image_nh.width

    cnt = offset # cnt gives offset for the ids of the masks

    qa_group = []  # list to collect all questions for current image

    for i_q_type, (q_type, q_template) in enumerate(question_templates.items()): # for each question type (inside)
        for i_class, (class_name, suffix) in enumerate({'hard exudate': 'EX'}.items()):  # Only for EX
            print("Image", test_image_index+1, ", questions about", class_name)

            # read OD mask from masks folder 
            od_gt = jp(config['path_data'], 'masks',  subset, 'OD', dirs.get_filename_without_extension(image_name_wo_ext) + '_' + 'OD' + '.' + config['format_gt'])
            od_img = Image.open(od_gt)
            od_img_np = np.array(od_img).astype(np.bool)

            # get OD diameter in original space
            od_diameter, _, _ = get_region_diameter(od_img_np)

            normal_question_counter = 0

            i_region = 0

            # get question
            question =  get_question_from_class(class_name, q_template, suffix)

            gt_location = jp(config['path_data'], 'masks',  subset, suffix, dirs.get_filename_without_extension(image_name_wo_ext) + '_' + suffix + '.' + config['format_gt'])

            while normal_question_counter < round(num_regions/2):
                mask_array = generate_circular_region_mask(np.array(ImageOps.grayscale(image_nh)), od_diameter)
                gt_mask = Image.open(gt_location)
                gt_mask_array = np.array(gt_mask)

                num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array)

                if num_pixels_in_region >= threshold and normal_question_counter < round(num_regions/2):
                    answer = 'yes'
                    mask_id = "mask" + str(cnt).zfill(6) + ".tif"

                    # generate resized masks and save them in corresponding folders
                    resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                    q_id = generate_question_id(i_class, i_q_type, i_region, test_image_index)
                    qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, center='random'))

                    normal_question_counter += 1
                    cnt += 1
                    i_region += 1       

                elif num_pixels_in_region == 0 and normal_question_counter < round(num_regions/2):   
                    answer = 'no'
                    mask_id = "mask" + str(cnt).zfill(6) + ".tif"               

                    # generate resized masks and save them in corresponding folders
                    resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                    q_id = generate_question_id(i_class, i_q_type, i_region, test_image_index)
                    qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, center='random'))
                    normal_question_counter += 1
                    cnt += 1
                    i_region += 1


            # generate two questions about regions centered at macula
            if image_name_wo_ext in available_fovea_centers:
                macula_loc = (int(fovea_centers[fovea_centers['image_id'] == image_name_wo_ext]['x']), int(fovea_centers[fovea_centers['image_id'] == image_name_wo_ext]['y']))
                # if I have the fovea center, generate two questions with center at macula and radius smaller than assumed OD diameter.
                for r in [0.5*od_diameter, 0.9*od_diameter]:   
                    mask_array = generate_circular_region_mask(np.array(ImageOps.grayscale(image_nh)), od_diameter, chosen_center=macula_loc, chosen_radius=r)         
                    gt_mask = Image.open(gt_location)
                    gt_mask_array = np.array(gt_mask)

                    num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array)

                    if num_pixels_in_region >= threshold:
                        answer = 'yes'
                        mask_id = "mask" + str(cnt).zfill(6) + ".tif"
                        role = 'sub'
                        # generate resized masks and save them in corresponding folders
                        resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                        q_id = generate_question_id(i_class, i_q_type, i_region, test_image_index)
                        qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, center='macula', role=role))

                        normal_question_counter += 1
                        cnt += 1
                        i_region += 1       

                    elif num_pixels_in_region == 0:   
                        answer = 'no'
                        mask_id = "mask" + str(cnt).zfill(6) + ".tif"               
                        role = 'sub'
                        # generate resized masks and save them in corresponding folders
                        resize_and_save(mask_array, resized_side, jp(path_output_mask, mask_id))

                        q_id = generate_question_id(i_class, i_q_type, i_region, test_image_index)
                        qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, center='macula', role=role))
                        normal_question_counter += 1
                        cnt += 1
                        i_region += 1
    return qa_group

def generate_dme_qa_single_grade_whole_fovea(path_image, img_index, offset, annotations_dme_grade, add_question_ex_in_fovea = False):
    # Generation of qa pair about DME grade for current image as determined by path_image
    cnt = offset
    qa_group = []

    # get image name from path
    image_name_wo_ext = dirs.get_filename_without_extension(path_image)
    image_name_w_ext = dirs.get_filename_with_extension(path_image)
    if '.' not in image_name_w_ext:
        image_name_w_ext = image_name_w_ext + '.jpg' # default format

    dme_grade = int(annotations_dme_grade[annotations_dme_grade['image_name'] == image_name_wo_ext]['dme_grade'])

    # DME grade question is simpler so I start with that one
    question = 'What is the diabetic macular edema grade for this image?'
    q_type = 'grade'
    answer = dme_grade
    mask_id = "whole_image_mask.tif" # use whole mask in this case. 
    role = 'main' # grade questions are always main
    # for this type of questions I will fix i_class, i_q_type and i_region. This will allow to make id unique.
    i_class = 10
    i_q_type = 10
    i_region = 0
    q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
    qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, role=role))

    # Whole question is also simple.
    question = 'Are there hard exudates in this image?'
    q_type = 'whole'
    if dme_grade == 0:
        answer = 'no'
    else:
        answer = 'yes'
    mask_id = "whole_image_mask.tif" # use whole mask in this case.         
    role = 'sub' # whole questions are treated as sub
    i_class = 12
    i_q_type = 12
    i_region = 0
    q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
    qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, role=role))

    # if required, as per config file, add fovea question about current image
    if add_question_ex_in_fovea:
        question = 'Are there hard exudates in the fovea?'
        q_type = 'fovea'
        if dme_grade == 2:
            answer = 'yes'
        else:
            answer = 'no'
        mask_id = "whole_image_mask.tif" # use whole mask in this case.         
        role = 'sub' # fovea questions are treated as sub
        i_class = 15
        i_q_type = 15
        i_region = 0
        q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
        qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, role=role))

    return qa_group

def generate_dme_qa_single_whole_test(image_name_wo_ext, img_index, offset, df_dme):
    # Generation of qa pair about DME grade for current image as determined by path_image
    qa_group = []

    # get image name from path
    image_name_w_ext = image_name_wo_ext + '.jpg'

    # DME grade question is simpler so I start with that one
    question = 'Are there hard exudates in this image?'
    q_type = 'whole'
    grade = int(df_dme[df_dme['image_name'] == image_name_wo_ext]['dme_grade'])
    if grade == 0: # depending on grade, define binary answer.
        answer = 'no'
    else:
        answer = 'yes'
    mask_id = "whole_image_mask.tif" # use whole mask in this case. 
    role = 'sub'
    # for this type of questions I will fix i_class, i_q_type and i_region. This will allow to make id unique.
    i_class = 12
    i_q_type = 12
    i_region = 0

    q_id = generate_question_id(i_class, i_q_type, i_region, img_index)
    qa_group.append(generate_dict_qa(image_name_w_ext, question, q_id, q_type, mask_id, answer, role=role))

    return qa_group, grade


def get_quadrants_answers(coords_h, coords_w, h_orig, w_orig, gt_location, resized_side, path_output_mask, threshold, path_img, question, i_class, i_q_type, q_type, img_index, suffix, healthy=False):
    # first, I have to determine for how many quadrants the answer is yes
    num_yes = 0
    quadrants_info = []
    i_region = 1000 # set as offset to prevent overlap with other masks
    cnt_mask = 0
    for i_h in range(len(coords_h)-1):
        for i_w  in range(len(coords_w)-1):
            top_left = (coords_h[i_h], coords_w[i_w])
            window_w = coords_w[i_w+1] - coords_w[i_w]
            window_h = coords_h[i_h+1] - coords_h[i_h]

            # build mask array
            mask_array = np.zeros((h_orig, w_orig), dtype = np.uint8)
            mask_array[top_left[0]:top_left[0]+window_h, top_left[1]:top_left[1]+window_w] = 1
            if healthy and suffix != 'FC': # if image is healthy, generate empty mask
                gt_mask_array = np.zeros((h_orig, w_orig), dtype=np.uint8)
                gt_mask = Image.fromarray(gt_mask_array)
            else:
                gt_mask = Image.open(gt_location)
                gt_mask_array = np.array(gt_mask)

            # if GT is almost completely white (object occupies almost whole picture), then just skip
            if np.count_nonzero(gt_mask_array) > 0.9*h_orig*w_orig:
                print("GT too big, too hard or impossible to find NO-questions. Skipping category")
                break

            if gt_mask_array.ndim > 2: # some gt images are RGBA in iDRID, which makes necessary to take only the red dimension
                gt_mask_array = gt_mask_array[:,:,0]  

            num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array)

            coords_norm = map_coords([top_left[1], top_left[1] + window_w, top_left[0], top_left[0]+window_h],
                                                (w_orig, h_orig), (resized_side, resized_side)) # [xmin, xmax, ymin, ymax] x means horizontal axis (dim 1), ymin is vertical axis (dim 0)

            mask_id = "maskq" + str(cnt_mask).zfill(6) + ".tif"

            # generate resized masks and save them in corresponding folders
            mask_to_save = np.zeros((resized_side, resized_side), dtype=np.uint8)
            mask_to_save[coords_norm[2]:coords_norm[3], coords_norm[0]:coords_norm[1]] = 255
            Image.fromarray(mask_to_save).save(jp(path_output_mask, mask_id)) # images are overwritten but it's okay because it's only 4 masks

            if healthy: # if healthy image, set alternative img_index
                q_id = int(str(i_class).zfill(2) + str(i_q_type).zfill(2) + str(i_region).zfill(4) + str(img_index+1111).zfill(4))
            else:
                q_id = int(str(i_class).zfill(2) + str(i_q_type).zfill(2) + str(i_region).zfill(4) + str(img_index).zfill(4))

            if num_pixels_in_region >= threshold:
                num_yes += 1
                quadrants_info.append({
                                'image_name': dirs.get_filename_with_extension(path_img),
                                'question': question,
                                'question_id': q_id,
                                'question_type': q_type,
                                'mask_name': mask_id,
                                'mask_coords': coords_norm,
                                'answer': 'yes'
            })
            else:
                quadrants_info.append({
                                'image_name': dirs.get_filename_with_extension(path_img),
                                'question': question,
                                'question_id': q_id,
                                'question_type': q_type,
                                'mask_name': mask_id,
                                'mask_coords': coords_norm,
                                'answer': 'no'
            })
            i_region += 1
            cnt_mask += 1

    return quadrants_info, num_yes


def append_qa_quadrant_single_balanced(config, subset, path_img, path_output_mask, path_healthy, img_index, offset):
    """Function to generate questions about the quadrants of an image for every biomarker. For each (image, biomarker), generate 2*(#num_yes_question) qa pairs
    meaning for each image I ask for example Are there hard exudates in this region? referring to all 4 quadrants. If 2 quadrants have EX, then I take all 4 quadrants.
    If only one quadrant has, then I take another one, if 4 quadrants have, then I get quadrants from a random healthy subject. This allows to balance the questions"""
    
    resized_side = config['size']
    question_templates = config['question_templates']    
    # if a threshold is specified, use it, otherwise use one pixel as threshold
    if 'threshold' in config:
        threshold = config['threshold'] #TODO: possible improvement, make threshold dependent on size of object (eg: 20% of pixels in object)
    else:
        threshold = 1

    # read the image
    image = Image.open(path_img)
    h_orig = image.height
    w_orig = image.width

    coords_h = np.linspace(0, h_orig, 3).astype(np.int)
    coords_w = np.linspace(0, w_orig, 3).astype(np.int)

    qa_group = []  # list to collect all questions for current image


    # list healthy images for current subset
    path_healthy_images = jp(path_healthy, subset)
    healthy_images = os.listdir(path_healthy_images)

    for i_q_type, (q_type, q_template) in enumerate(question_templates.items()): # for each question type (inside)
        for i_class, (class_name, suffix) in enumerate(config['map_class_suffix'].items()):  # for each abnormality
            print("Image", img_index+1, ", questions about", class_name)
            # check if class is present in the current image
            gt_location = jp(config['path_data'], 'masks',  subset, suffix, dirs.get_filename_without_extension(path_img) + '_' + suffix + '.' + config['format_gt'])
            if not os.path.exists(gt_location):
                print("No GT, skipping for this category...")
                continue # continue if gt does not exist for this patients because all questions would be answered with NO, contributing to the imbalance

            question =  get_question_from_class(class_name, q_template, suffix)

            quadrants_info, num_yes = get_quadrants_answers(coords_h, coords_w, h_orig, w_orig, gt_location, resized_side, path_output_mask, threshold, path_img, question, i_class, i_q_type, q_type, img_index, suffix)

            if num_yes == 0:
                # if biomarker not present in any of the quadrants, skip this biomarker
                continue
            elif num_yes == 1:
                # if only one quadrant has it, take that one and another one (randomly chosen). For macula it should always come here
                no_ind = []
                yes_ind = None
                for i_e, e in enumerate(quadrants_info):
                    if e['answer'] == 'yes':
                        yes_ind = i_e
                    else:
                        no_ind.append(i_e)
                qa_group.append(quadrants_info[yes_ind])
                qa_group.append(quadrants_info[random.choice(no_ind)])
            elif num_yes == 2:
                # if two, take all 4 quadrants as samples
                qa_group += quadrants_info
            elif num_yes == 3: 
                # if three, take them, take the remaining, and generate two more from healthy images
                qa_group += quadrants_info
                # now take a random healthy image and extract questions about two quadrants
                num_yes_repl = 4
                while num_yes_repl >= 3: # I need num_yes < 3, so that I have at least 2 no-answer
                    healthy_random = random.choice(healthy_images)
                    path_random = jp(path_healthy_images, healthy_random)
                    image_repl = Image.open(path_random)
                    h_orig_repl = image_repl.height
                    w_orig_repl = image_repl.width                
                    coords_h_repl = np.linspace(0, h_orig_repl, 3).astype(np.int)
                    coords_w_repl = np.linspace(0, w_orig_repl, 3).astype(np.int)

                    gt_location_repl = jp(config['path_data'], 'masks',  subset, suffix, dirs.get_filename_without_extension(path_random) + '_' + suffix + '.' + config['format_gt'])

                    # build plurals for text questions
                    question = get_question_from_class(class_name, q_template, suffix)

                    quadrants_info_repl, num_yes_repl = get_quadrants_answers(coords_h_repl, coords_w_repl, h_orig_repl, w_orig_repl, gt_location_repl, resized_side, path_output_mask, threshold, path_random, question, i_class, i_q_type, q_type, img_index, suffix, healthy=True)

                if num_yes_repl == 2: # if answers are yes for two quadrants, take the other two to compensate
                    qa_group += [e for e in quadrants_info_repl if e['answer'] == 'no']
                else: 
                    # if only one or zero yes, choose two of the quadrants randomly
                    neg = [e for e in quadrants_info_repl if e['answer'] == 'no']
                    random.shuffle(neg)
                    qa_group += neg[:2]
                # normalize image and copy it to output visual 
                ip.normalize_and_save(path_random, jp(path_output_mask[:path_output_mask.find('masks')], 'visual', subset, healthy_random), resize=config['resize'], size = config['size'], normalize=config['normalize'])                
            else:
                # if four (can't be more), take all of them, and use all quadrants of a healthy image
                qa_group += quadrants_info
                # now take a random healthy image and extract questions about all quadrants
                num_yes_repl = 4 # assume all quadrants
                while num_yes_repl > 0: # I need num_yes < 3, so that I have at least 2 no-answer
                    healthy_random = random.choice(healthy_images)
                    path_random = jp(path_healthy_images, healthy_random)
                    image_repl = Image.open(path_random)
                    h_orig_repl = image_repl.height
                    w_orig_repl = image_repl.width                
                    coords_h_repl = np.linspace(0, h_orig_repl, 3).astype(np.int)
                    coords_w_repl = np.linspace(0, w_orig_repl, 3).astype(np.int)

                    gt_location_repl = jp(config['path_data'], 'masks',  subset, suffix, dirs.get_filename_without_extension(path_random) + '_' + suffix + '.' + config['format_gt'])

                    # build plurals for text questions
                    question = get_question_from_class(class_name, q_template, suffix)

                    quadrants_info_repl, num_yes_repl = get_quadrants_answers(coords_h_repl, coords_w_repl, h_orig_repl, w_orig_repl, gt_location_repl, resized_side, path_output_mask, threshold, path_random, question, i_class, i_q_type, q_type, img_index, suffix, healthy=True)

                qa_group += quadrants_info_repl

                # normalize image and copy it to output visual 
                ip.normalize_and_save(path_random, jp(path_output_mask[:path_output_mask.find('masks')], 'visual', subset, healthy_random), resize=config['resize'], size = config['size'], normalize=config['normalize'])

    return qa_group


def append_qa_quadrant_single(config, subset, path_img, path_output_mask, path_healthy, img_index):
    """Same as append_qa_quadrant_single_balanced, except I don't balance the questions. For each (image, biomarker) I generate 4 qa pairs and I don't care about 
    the answer because these questions are for the test set."""

    resized_side = config['size']
    question_templates = config['question_templates']    
    # if a threshold is specified, use it, otherwise use one pixel as threshold
    if 'threshold' in config:
        threshold = config['threshold'] #TODO: possible improvement, make threshold dependent on size of object (eg: 20% of pixels in object)
    else:
        threshold = 1

    # read the image
    image = Image.open(path_img)
    h_orig = image.height
    w_orig = image.width

    coords_h = np.linspace(0, h_orig, 3).astype(np.int)
    coords_w = np.linspace(0, w_orig, 3).astype(np.int)

    qa_group = []  # list to collect all questions for current image


    # list healthy images for current subset
    path_healthy_images = jp(path_healthy, subset)
    healthy_images = os.listdir(path_healthy_images)

    for i_q_type, (q_type, q_template) in enumerate(question_templates.items()): # for each question type (inside)
        for i_class, (class_name, suffix) in enumerate(config['map_class_suffix'].items()):  # for each abnormality
            print("Image", img_index+1, ", questions about", class_name)
            # check if class is present in the current image
            gt_location = jp(config['path_data'], 'masks',  subset, suffix, dirs.get_filename_without_extension(path_img) + '_' + suffix + '.' + config['format_gt'])
            if not os.path.exists(gt_location):
                print("No GT, skipping for this category...")
                continue # continue if gt does not exist for this patients because all questions would be answered with NO, contributing to the imbalance

            question =  get_question_from_class(class_name, q_template, suffix)

            quadrants_info, num_yes = get_quadrants_answers(coords_h, coords_w, h_orig, w_orig, gt_location, resized_side, path_output_mask, threshold, path_img, question, i_class, i_q_type, q_type, img_index, suffix)
    
            # in this case I don't balance so I just have to add the questions about all 4 quadrants independently of the answer
            qa_group += quadrants_info
    
    return qa_group




def generate_qa_complement(config, subset, path_img, path_output_masks_A, path_output_masks_B, img_index):
    # generation of questions based on the inside and outside of a window for image in path_image according to information in config
    num_regions = config['num_regions']
    max_window_side_resized = config['max_window_side']
    min_window_side_resized = config['min_window_side']
    prop = config['proportion_deviation']
    resized_side = config['size']
    question_templates = config['question_templates']

    # read the image
    image = Image.open(path_img)
    h_orig = image.height
    w_orig = image.width

    # convert resized sizes to original dimensions
    max_window_width_orig = round((w_orig/resized_side)*max_window_side_resized)
    min_window_width_orig = round((w_orig/resized_side)*min_window_side_resized)
    max_window_height_orig = round((h_orig/resized_side)*max_window_side_resized)
    min_window_height_orig = round((h_orig/resized_side)*min_window_side_resized)

    cnt = num_regions*img_index # cnt gives offset for the ids of the masks

    qa_group = []  # list to collect all questions for current image

    for i_region in tqdm(range(num_regions)): # for each of the desired windows
        for i_q_type, (q_type, q_template) in enumerate(question_templates.items()):
            mask_id = cnt
            top_left, window_w, window_h = generate_random_window(w_orig, h_orig, min_window_width_orig, max_window_width_orig, min_window_height_orig, max_window_height_orig, prop, resized_side)
            
            # build mask array
            mask_array = np.zeros((h_orig, w_orig), dtype = np.uint8)
            mask_array[top_left[0]:top_left[0]+window_h, top_left[1]:top_left[1]+window_w] = 1
            # build complementary mask array
            mask_array_complement = np.ones_like(mask_array) # generate mask for complement mask
            mask_array_complement[mask_array>0] = 0

            # define name for masks to be saved
            mask_id = "mask" + str(cnt).zfill(6) + ".tif"
            compl_mask_id = "cmask" + str(cnt).zfill(6) + ".tif"

            coords_norm = map_coords([top_left[1], top_left[1] + window_w, top_left[0], top_left[0]+window_h],
                                                            (w_orig, h_orig), (resized_side, resized_side)) # [xmin, xmax, ymin, ymax]

            # generate resized masks and save them in corresponding folders
            mask_to_save = np.zeros((resized_side, resized_side), dtype=np.uint8)
            mask_to_save[coords_norm[2]:coords_norm[3], coords_norm[0]:coords_norm[1]] = 255
            Image.fromarray(mask_to_save).save(jp(path_output_masks_A, mask_id))
            compl_mask_to_save = 255*np.ones_like(mask_to_save, dtype=np.uint8) # generate mask for complement mask
            compl_mask_to_save[coords_norm[2]:coords_norm[3], coords_norm[0]:coords_norm[1]] = 0
            Image.fromarray(compl_mask_to_save).save(jp(path_output_masks_B, compl_mask_id))

            # iterate through classes and get answers
            gt_available = True
            for i_class, (class_name, suffix) in enumerate(config['map_class_suffix'].items()): 
                question = q_template.replace('<>', class_name+'s') + '?'
                gt_location = jp(config['path_data'], 'masks',  subset, suffix, dirs.get_filename_without_extension(path_img) + '_' + suffix + '.' + config['format_gt'])
                if not os.path.exists(gt_location):
                    answer = 'no'
                    gt_available = False

                if gt_available:
                    gt_mask = Image.open(gt_location)
                    gt_mask_array = np.array(gt_mask)

                    if gt_mask_array.ndim > 2: # some gt images are RGBA, which makes necessary to take only the red dimension
                        gt_mask_array = gt_mask_array[:,:,0]  

                    # depending on question type, answer based on the inside mask or the outside mask
                    if q_type == 'inside':
                        num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array)
                    elif q_type == 'outside':
                        num_pixels_in_region = np.count_nonzero(gt_mask_array*mask_array_complement)

                    if num_pixels_in_region>0:
                        answer = 'yes'
                    else:
                        answer = 'no'

                qa_group.append({
                                    'image_name': dirs.get_filename_with_extension(path_img),
                                    'question': question,
                                    'question_id': int(str(i_class).zfill(2) + str(i_q_type).zfill(2) + str(i_region).zfill(4) + str(img_index).zfill(4)),
                                    'question_type': q_type,
                                    'maskA_name': mask_id,
                                    'maskB_name': compl_mask_id,
                                    'mask_coords': coords_norm,
                                    'answer': answer
                })
        cnt += 1

    return qa_group

def generate_qa_dual(config, subset, path_img, img_index):
    # generation of questions based on two separate windows for image in path_image according to information in config
    # TODO
    return 34