## How  to set up the environment

The environment needed is the same as MMBench



## How to choose one stage for the test

change the "stage" in Supervised_Learning_2, line 92.

stage=1 means that we only test "encoder" stage;

stage=2 fusion

stage=3 head

stage=4 all 3 stages



## How to run the code

after choosing the stage, run the following codes for 3 different types of test:

**1.nsight compute**

`sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__inst_executed.avg.per_cycle_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct /home/user/anaconda3/envs/multibench/bin/python  /home/user/MMBench/applications/Sarcasm/NVIDIA_test/affect_early_fusion.py >early.csv`

**2.nsight system**

`/usr/local/cuda-11.6/bin/nsys profile --stats=true python /home/user/MMBench/applications/Sarcasm/NVIDIA_test/affect_early_fusion.py`

**3.pytorch profiler** 

`python /home/user/MMBench/applications/Sarcasm/pytorch_profiler_test/affect_early_profiler.py`

