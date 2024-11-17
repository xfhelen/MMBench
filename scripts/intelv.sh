#!/bin/bash

# 使用示例 'scripts/intelv.sh' 'applications/Customization/cust.py' --path_config 'applications/Customization/task6.yaml' --options normal

PYTHON=$(which python)

PYTHON_SCRIPT="$1"
PYTHON_ARGS="${@:2}"

rm -rf ./vtune

echo "${PYTHON} ${PYTHON_SCRIPT} ${PYTHON_ARGS} --intelv" > "temp_file1.txt"
vtune -collect hotspots -result-dir "./vtune" ${PYTHON} ${PYTHON_SCRIPT} ${PYTHON_ARGS} --intelv  >> "temp_file1.txt"

echo "IPC" >> "temp_file1.txt"
vtune -report hw-events  -r ./vtune -format=csv --csv-delimiter=comma > "temp_file2.csv"

echo " aten::" >> "temp_file1.txt"
vtune -report hotspots  -r ./vtune -group-by=task -format=csv --csv-delimiter=comma > "temp_file3.csv"

python scripts/intelv.py 

rm -rf ./vtune
rm -rf temp_file1.txt
rm -rf temp_file2.csv
rm -rf temp_file3.csv
