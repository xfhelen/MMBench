#!/bin/bash

nsys profile --stats=true python "$1" > "temp_file1.txt"

rm report*.nsys-rep
rm report*.sqlite