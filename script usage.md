## 运行ncu脚本
该脚本调用ncu来测量部分硬件使用情况。脚本的第一个参数为python代码的路径，第二个参数为输出文件的路径。同时该脚本可以实现对需要的硬件使用情况的平均值求取与输出。

使用示例：
```bash
./metric.sh applications/Vison\&Touch/LRTF.py applications/Vison\&Touch/ncu_info.csv
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