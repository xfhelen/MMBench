#!/bin/bash

# 默认情况下，使用系统中的 Python 程序
PYTHON="/home/zhuxiaozhi/anaconda3/envs/multibench/bin/python"

# 处理命令行参数
while getopts ":p:" opt; do
  case ${opt} in
    p )
      PYTHON="$OPTARG"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Option -$OPTARG requires an argument." 1>&2
      exit 1
      ;;
  esac
done

shift $((OPTIND -1))

# 运行命令
sudo PYTHONPATH="/home/zhuxiaozhi/MMBench" /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__inst_executed.avg.per_cycle_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,sm__warps_active.avg.pct_of_peak_sustained_active "$PYTHON" "$1" > "$2"

