#!/bin/bash

PYTHON=$(which python)
PYTHON_ARGS="${@:2}"

sudo /usr/local/cuda-11.7/bin/nsys profile --stats=true $PYTHON "$1" $PYTHON_ARGS > "scripts/nsys_temp_file.txt"

sudo rm report*.nsys-rep
sudo rm report*.sqlite