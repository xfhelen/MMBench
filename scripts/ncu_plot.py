import matplotlib.pyplot as plt
import numpy as np
def read_metrics_from_file(file_path):
    metrics = {
        'DRAM utilization': [],
        'achieved occupancy': [],
        'IPC': [],
        'GLD efficiency (global load efficiency)': [],
        'GST efficiency (global store efficiency)': []
    }
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Metric:'):
                parts = line.split(',')
                metric_name = parts[0].split(': ')[1]
                average_value = float(parts[1].split(': ')[1])
                if metric_name in metrics:
                    metrics[metric_name].append(average_value)
    
    return metrics

def plot_metrics(metrics):
    metric_names = list(metrics.keys())
    num_groups = len(metrics[metric_names[0]])
    num_metrics = len(metric_names)
    
    # 放大 IPC 数据
    if 'IPC' in metric_names:
        metrics['IPC'] = [v * 10 for v in metrics['IPC']]
    
    # 设置柱状图的宽度和位置
    bar_width = 0.15
    index = np.arange(num_metrics)
    
    plt.figure(figsize=(12, 8))
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for group_idx in range(num_groups):
        values = [metrics[metric][group_idx] for metric in metric_names]
        plt.bar(index + group_idx * bar_width, values, bar_width, label=f'Group {group_idx + 1}', color=colors[group_idx % len(colors)])
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Metrics Bar Plot')
    plt.xticks(index + bar_width * (num_groups - 1) / 2, metric_names, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig("test.png")

# 示例使用
file_path = 'scripts/metrics_output.txt'
metrics = read_metrics_from_file(file_path)
plot_metrics(metrics)