# -*- coding: utf-8 -*-
import csv
import sys

if len(sys.argv) != 4:
    print("Usage: python script.py input_file_path output_file_path")
    sys.exit(1)

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
description = sys.argv[3]

print("Output file path: {}".format(output_file_path))

# 打开并读取输入文件
with open(input_file_path, 'r') as file:
    data = file.read()

# 提取所需数据
lines = data.split('\n')
extracted_data = []
for line in lines:
    if line.strip().startswith(('dram_', 'smsp_', 'sm_', 'gpu_', 'stall_', 'achieved_', 'ipc', 'gld_', 'gst_')):
        extracted_data.append(line.strip().split())

# 创建CSV文件并写入数据
with open(output_file_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Unit', 'Value'])
    for item in extracted_data:
        writer.writerow(item)

# 收集和计算平均值
metric_data = {}
count_data = {}
for item in extracted_data:
    metric = item[0]
    value = float(item[2])
    if metric in metric_data:
        metric_data[metric] += value
        count_data[metric] += 1
    else:
        metric_data[metric] = value
        count_data[metric] = 1

# 计算平均值并打印结果
for metric, total_value in metric_data.items():
    count = count_data[metric]
    average_value = total_value / count
    if metric == 'dram__throughput.avg.pct_of_peak_sustained_elapsed':
        metric_name = 'DRAM utilization'
    if metric == 'sm__warps_active.avg.pct_of_peak_sustained_active':
        metric_name = 'achieved occupancy'
    if metric == 'smsp__inst_executed.avg.per_cycle_active':
        metric_name = 'IPC'
    if metric == 'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct':
        metric_name = 'GLD efficiency (global load efficiency)'
    if metric == 'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct':
        metric_name = 'GST efficiency (global store efficiency)'
    print("Metric: {}, Average Value: {}".format(metric_name, average_value))

import os 
file_path = 'scripts/metrics_output.txt'

# 打开文件，追加模式，如果文件不存在则创建
with open(file_path, 'a') as file:
    # 写入一个空行，分隔不同次添加
    if os.path.getsize(file_path) > 0:
        file.write('\n')
    file.write("{}\n".format(description))
    
    for metric, total_value in metric_data.items():
        count = count_data[metric]
        average_value = total_value / count
        if metric == 'dram__throughput.avg.pct_of_peak_sustained_elapsed':
            metric_name = 'DRAM utilization'
        elif metric == 'sm__warps_active.avg.pct_of_peak_sustained_active':
            metric_name = 'achieved occupancy'
        elif metric == 'smsp__inst_executed.avg.per_cycle_active':
            metric_name = 'IPC'
        elif metric == 'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct':
            metric_name = 'GLD efficiency (global load efficiency)'
        elif metric == 'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct':
            metric_name = 'GST efficiency (global store efficiency)'
        else:
            metric_name = metric  # 如果没有匹配的名称，使用原始名称

        # 写入文件
        
        file.write("Metric: {}, Average Value: {}\n".format(metric_name, average_value))