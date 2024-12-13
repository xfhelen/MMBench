#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
options=("normal" "encoder" "fusion" "head")
step=("type2")
for i in "${options[@]}"
do
    for j in "${step[@]}"
    do
        echo "option: $i step: $j" 
        echo "option: $i step: $j" >> "nsys_result.txt"
        echo "option: $i step: $j" >> "ncu_result.txt"
        ./scripts/nsys_metric.sh applications/Customization/cust.py --options "$i" --path_config "applications/Customization/$j.yaml" > /dev/null 2>&1
        python scripts/nsys_visualization.py >> "nsys_result.txt"
        mv pie.html "pie_${i}_${j}.html"
        ./scripts/ncu_metric.sh applications/Customization/cust.py "ncu_${i}_${j}.csv" x --options "$i" --path_config "applications/Customization/$j.yaml" >> ncu_result.txt
    done
done