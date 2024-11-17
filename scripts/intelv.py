import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.commons.utils import JsCode

print("----------------------------------\n")

# 读取temp_file2.csv计算IPC
df_perf = pd.read_csv('temp_file2.csv')
instructions = df_perf['Hardware Event Count:INST_RETIRED.ANY'].iloc[0]
cycles = df_perf['Hardware Event Count:CPU_CLK_UNHALTED.THREAD'].iloc[0]
ipc = instructions / cycles

# 从temp_file1.txt读取CPU占用率
with open('temp_file1.txt', 'r') as f:
    content = f.read()
    # 使用字符串查找提取CPU利用率
    cpu_util_line = [line for line in content.split('\n') if 'Effective CPU Utilization:' in line][0]
    cpu_utilization = float(cpu_util_line.split(':')[1].strip().rstrip('%'))

print(f"IPC: {ipc:.2f}")
print(f"CPU Utilization: {cpu_utilization:.2f}%")

print("\n")

# 读取temp_file3.cs计算算子时间占比
df = pd.read_csv('temp_file3.csv')

# 移除[Outside any task]行
df = df[df['Task Type'] != '[Outside any task]']

# 定义算子类别
categories = {
    'activation': ['relu', 'gelu', 'clamp_min', 'softmax'],  
    'conv': ['convolution', '_convolution', 'conv1d', 'conv2d', 'mkldnn_convolution'],
    'bnorm': ['batch_norm', '_batch_norm_impl_index', 'layer_norm', 'native_layer_norm'],
    'elewise': ['add_', 'add', 'div', 'sub', 'eltwise'],
    'pooling': ['pool2d', 'adaptive_avg_pool2d'],
    'gemm': ['linear', 'addmm', 'matmul', 'mul', 'bmm'],
    'reduce': ['sum', 'mean'],
    'other': []  # 其他所有操作
}

# 初始化每个类别的总时间
time_stats = {cat: 0.0 for cat in categories.keys()}

# 计算每个类别的总CPU时间
for index, row in df.iterrows():
    task_name = row['Task Type']
    cpu_time = row['CPU Time'] if pd.notna(row['CPU Time']) else 0.0
    
    # 检查任务属于哪个类别
    categorized = False
    for cat, keywords in categories.items():
        if any(keyword in task_name.lower() for keyword in keywords):
            time_stats[cat] += cpu_time
            categorized = True
            break
    
    # 如果没有匹配到任何类别，归入other类
    if not categorized and cpu_time > 0:
        time_stats['other'] += cpu_time

# 计算百分比
total_time = sum(time_stats.values())
percentages = [(k, (v/total_time)*100) for k, v in time_stats.items()]

# 打印原始数据用于调试
print("Raw Time Usage(ms):")
for cat, time in time_stats.items():
    print(f"{cat}: {time:.3f}")

print("\nPercentage Statistics:")
for cat, pct in percentages:
    print(f"{cat}: {pct:.2f}%")

# 创建饼图
pie = (
    Pie()
    .add(
        "算子时间占比",
        percentages,
        radius=["40%", "75%"],
    )
    .set_colors(["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272", "#fc8452", "#9a60b4"])
    .set_global_opts(
        title_opts=opts.TitleOpts(title="算子CPU时间占比分析"),
        legend_opts=opts.LegendOpts(orient="vertical", pos_top="middle", pos_left="right")
    )
    .set_series_opts(
        label_opts=opts.LabelOpts(
            formatter=JsCode(
                "function(params) {return params.name + ': ' + params.value.toFixed(2) + '%';}"
            )
        )
    )
)

# 保存图表
pie.render("pie_intelv.html")


