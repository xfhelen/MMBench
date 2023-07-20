# Project:
#   VQA
# Description:
#   Functions for printing stuff
# Author: 
#   Sergio Tascon-Morales


def print_section(section_name):
    print(40*"~")
    print(section_name)
    print(40*"~")

def print_line():
    print(40*'-')

def print_event(text):
    print('-> Now doing:', text, '...')