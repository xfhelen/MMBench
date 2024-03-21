#!/bin/bash

# 默认情况下，使用系统中的 Python 程序
PYTHON=$(which python)

# 保存传递给脚本的参数
PYTHON_SCRIPT="$1"
PYTHON_ARGS="${@:3}"

# 运行命令并将输出重定向到临时文件（文件1）
sudo /usr/local/cuda-*/nsight-compute-*/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__inst_executed.avg.per_cycle_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,sm__warps_active.avg.pct_of_peak_sustained_active "$PYTHON" "$PYTHON_SCRIPT" $PYTHON_ARGS > "temp_file1.txt"

# 使用python命令行并将文件1作为输入，同时将输出重定向到文件2
sudo "$PYTHON" scripts/ncu_dataprocess.py "temp_file1.txt" "$2"

# 删除临时文件（文件1）
rm "temp_file1.txt"


