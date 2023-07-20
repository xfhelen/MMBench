# Project:
#   VQA
# Description:
#   Functions for folder creation, etc
# Author: 
#   Sergio Tascon-Morales

import os
from os.path import join as jp
import ntpath
import shutil
 
def create_folder(path):
    # creation of folders
    if not os.path.exists(path):
        try:
            os.mkdir(path) # try to create folder
        except:
            os.makedirs(path) # create full path

def is_empty(path):
    return len(os.listdir(path)) < 1

def clean_folder(path):
    if is_empty(path): # already empty
        return
    else:
        files_and_folders = os.listdir(path)
        for elem in files_and_folders:
            if os.path.isdir(jp(path,elem)):
                shutil.rmtree(jp(path, elem))
            else:
                os.remove(jp(path,elem))

def list_folders(path): 
    # lists folders only in path
    return [k for k in os.listdir(path) if os.path.isdir(jp(path, k))]

def list_files(path):
    return [k for k in os.listdir(path) if not os.path.isdir(jp(path, k))]

def create_folders_within_folder(parent_path, folder_name_list):
    # create folders and return paths
    paths = []
    for folder_name in folder_name_list:
        folder_path = jp(parent_path, folder_name)
        create_folder(folder_path)
        paths.append(folder_path)
    return paths

def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]

def get_filename_with_extension(path):
    filename = get_filename(path)
    return filename

def remove_whole_folder(path):
    shutil.rmtree(path)