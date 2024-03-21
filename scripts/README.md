## Run Nsight Compute(ncu) script
This script uses ncu to measure hardware utilization. The first argument of the script is the path to the Python code, and the second argument is the path to the output file. The script can also calculate and output the average values of the desired hardware metrics.

Please make sure to replace the Python path in the script with your own before using.

### Usage Example
(Note that the working directory is MMBench):
```bash
./scripts/ncu_metric.sh applications/Vision-Touch/LRTF.py applications/Vision-Touch/ncu_info.csv
```

Command line output:
```bash
Output file path: applications/Vision-Touch/ncu_info.csv
Metric: DRAM utilization, Average Value: 17.131235955056173
Metric: achieved occupancy, Average Value: 37.859348314606784
Metric: IPC, Average Value: 0.1575730337078654
Metric: GLD efficiency (global load efficiency), Average Value: 68.62361797752806
Metric: GST efficiency (global store efficiency), Average Value: 89.53728089887643
```

There are also optional parameters (normal/encoder/fusion/head, with normal being the default) for measuring in different stages:
```bash
./scripts/ncu_metric.sh applications/Vision-Touch/LRTF.py applications/Vision-Touch/ncu_info_encoder.csv --options encoder
```

## Run Nsight System(sys) script
This script uses nsys to measure GPU data. The argument of the script is the Python code to be executed.

### Usage Example
(Note that the working directory is MMBench):

```bash
./scripts/nsys_metric.sh applications/Vision-Touch/LRTF.py
```
All the output data is stored in the `scripts/nsys_temp_file.txt` file.

## 运行ncu脚本
该脚本调用ncu来测量部分硬件使用情况。脚本的第一个参数为python代码的路径，第二个参数为输出文件的路径。同时该脚本可以实现对需要的硬件使用情况的平均值求取与输出。

使用前请注意将脚本中的python路径换成自己的。

### 使用示例
(请注意工作目录为MMBench)：
```bash
./scripts/ncu_metric.sh applications/Vision-Touch/LRTF.py applications/Vision-Touch/ncu_info.csv
```

命令行输出情况：
```bash
Output file path: applications/Vision-Touch/ncu_info.csv
Metric: DRAM utilization, Average Value: 17.131235955056173
Metric: achieved occupancy, Average Value: 37.859348314606784
Metric: IPC, Average Value: 0.1575730337078654
Metric: GLD efficiency (global load efficiency), Average Value: 68.62361797752806
Metric: GST efficiency (global store efficiency), Average Value: 89.53728089887643
```

同时还有可选参数(normal/encoder/fusion/head，其中默认为normal)，能够实现分阶段测量：
```bash
./scripts/ncu_metric.sh applications/Vision-Touch/LRTF.py applications/Vision-Touch/ncu_info_encoder.csv --options encoder
```

## nsys脚本
该脚本调用nsys来测量GPU数据。脚本的参数为需要运行的python代码

### 使用示例
(请注意工作目录为MMBench)：
```bash
./scripts/nsys_metric.sh applications/Vision-Touch/LRTF.py
```

输出的全部数据存在 `scripts/nsys_temp_file.txt` 文件中。
