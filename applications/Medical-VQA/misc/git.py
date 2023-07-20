# Project:
#   VQA
# Description:
#   GIT utils
# Author: 
#   Sergio Tascon-Morales

import os
import subprocess

def get_commit_hash():
    """Function to get the commit hash.

    Returns
    -------
    str
        Commit hash of this version of the code
    """
    old_path = os.getcwd() 
    os.chdir(os.path.dirname(os.path.abspath(__file__))) # enter path of train file which is repo folder
    try:
        h = subprocess.check_output(["git", "log", "--pretty=format:%H", "-n", "1"]).decode()
    except:
        h = "UnknownHash"
    os.chdir(old_path)
    return h