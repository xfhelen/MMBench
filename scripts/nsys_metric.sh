#!/bin/bash

nsys profile --stats=true python "$1" > "scripts/nsys_temp_fil e.txt"

rm report*.nsys-rep
rm report*.sqlite