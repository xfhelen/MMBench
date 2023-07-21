## 运行ncu脚本
该脚本调用ncu来测量部分硬件使用情况。脚本的第一个参数为python代码的路径，第二个参数为输出文件的路径。同时该脚本可以实现对需要的硬件使用情况的平均值求取与输出。

使用示例(请注意工作目录为MMBench)：
```bash
./scripts/ncu_metric.sh applications/Vison\&Touch/LRTF.py applications/Vison\&Touch/ncu_info.csv
```

命令行输出情况：
```bash
Input file path: temp_file1.txt
Output file path: applications/Vison&Touch/ncu_info.csv
Metric: dram__throughput.avg.pct_of_peak_sustained_elapsed, Average Value: 16.996696629213485
Metric: sm__warps_active.avg.pct_of_peak_sustained_active, Average Value: 37.9523146067416
Metric: smsp__inst_executed.avg.per_cycle_active, Average Value: 0.15584269662921366
Metric: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct, Average Value: 68.62361797752806
Metric: smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct, Average Value: 89.53728089887643
```

同时还有可选参数(normal/encoder/fusion/head，其中默认为normal)，能够实现分阶段测量：
```bash
./scripts/ncu_metric.sh applications/Vison\&Touch/LRTF.py applications/Vison\&Touch/ncu_info_encoder.csv --options encoder
```

## nsys脚本
该脚本调用ncu来测量GPU数据。脚本的参数为需要运行的python代码

使用示例(请注意工作目录为MMBench)：
```bash
./scripts/nsys_metric.sh applications/Vison\&Touch/LRTF.py
```

输出的内容暂时存在scripts目录下的temp_file1.txt文件中。