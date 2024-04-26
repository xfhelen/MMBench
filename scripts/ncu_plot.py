import matplotlib.pyplot as plt

# 数据
metrics = ["DRAM utilization", "achieved occupancy", "GLD efficiency", "GST efficiency", "IPC"]
average_values = [17.131235955056173, 37.859348314606784, 68.62361797752806, 89.53728089887643, 0.1575730337078654]

# 设置 IPC 的颜色和范围
ipc_color = '#87CEEB'  # 湖蓝色
ipc_values = [average_values[-1]]  # 只显示 IPC 的值
ipc_ticks = [0, 5]  # y 轴刻度范围

# 设置其他指标的颜色和范围
red_palette = ['#FFB6C1', '#FFA07A', '#FA8072', '#FF6347']  # 红色系
other_values = average_values[:-1]
other_ticks = [0, 50, 100]  # y 轴刻度范围

# 创建图形和子图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制左边 y 轴的柱状图
ax1.bar(metrics[:-1], other_values, color=red_palette)
ax1.set_ylabel('Average Value', color=red_palette[0])
ax1.tick_params(axis='y', labelcolor=red_palette[0])
ax1.set_yticks(other_ticks)

# 创建右边 y 轴
ax2 = ax1.twinx()

# 绘制右边 y 轴的柱状图
ax2.bar(metrics[-1:], ipc_values, color=ipc_color)  # IPC 是最后一个指标
ax2.set_ylabel('IPC', color=ipc_color)
ax2.tick_params(axis='y', labelcolor=ipc_color)
ax2.set_ylim(ipc_ticks)

# 添加标题
plt.title('Average Values and IPC')

# 展示图形
plt.show()


