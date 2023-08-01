# -*- coding: utf-8 -*-
import csv
import sys

if len(sys.argv) != 3:
    print("Usage: python script.py input_file_path output_file_path")
    sys.exit(1)

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

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
