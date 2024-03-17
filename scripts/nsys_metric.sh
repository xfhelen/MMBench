#!/bin/bash

PYTHON=$(which python)
sudo PYTHONPATH="./" /usr/local/cuda-*/bin/nsys profile --stats=true $PYTHON "$1" > "scripts/nsys_temp_file.txt"

sudo rm report*.nsys-rep
sudo rm report*.sqlite