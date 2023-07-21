#!/bin/bash

nsys profile --stats=true python "$1" > "scripts/temp_file1.txt"

rm report*.nsys-rep
rm report*.sqlite