## set up

`git clone https://github.com/xfhelen/MMBench.git`

`cd applications/TransFuser/transfuser-2022`

`chmod +x setup_carla.sh`

`./setup_carla.sh`

`conda env create -f environment.yml`

`conda activate tfuse`

`pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html`

`pip install mmcv-full==1.5.3 -f*`https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html`

## Dataset

You can download the dataset (210GB) by running:

`chmod +x download_data.sh`

`./download_data.sh`

## Test

there are 3 stages(encoder fusion head). it is controlled by “stage”

(which is located at ~/team_code_transfuser/transfuser.py, line 136). 

stage is used to analyze 3 stages (encoders fusion head),change it to test different stages. 

stage=1 : encoder

stage=2 : fusion

stage=3 : head

stage=4 : all 3 stages used(normal conditions)

we provide 3 types of tests, which are (1)ncu, (2)nsys and (3)pytorch profiler. 

**1. nsight compute**

to use nsight compute(ncu) test,follow the 3 steps below:

*(1)change “stage” to what you want to test.* 

*(2)cd /team_code_transfuser.* 

*(3)sudo (path to ncu)/ncu --metrics (parameters) (path to python/python3) python*

*Nvidia_train.py (parameter) .* 

and here is an example to use ncu test:



`sudo` `/usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__inst_executed.avg.per_cycle_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,__sm__warps_active.avg.pct_of_peak_sustained_active /home/tangpeng/miniconda3/envs/tfuse/bin/python Nvidia_train.py --batch_size 10 --logdir` `/home/niumo/transfuser-2022/logdir` `--root_dir` `/home/tangpeng/transfuser/data/``--parallel_training 0 --epoch 10 > transfuser_1.csv`



**2. nsight systems**

*(1)change “stage” to what you want to test.* 

*(2)cd /team_code_transfuser.* 

*(3)(path to nsys) nsys profile --stats=true (path to python) python Nvidia_train.py*

*(parameter)*

here is an example to use nsys:

`/usr/local/cuda-11.6/bin/nsys profile --stats=true python Nvidia_train.py --batch_size 10`

`--logdir /home/niumo/transfuser-2022/logdir --root_dir /home/tangpeng/transfuser/data/`

`--parallel_training 0 --epoch 1`



**3. pytorch profiler**

*(1)change “stage”*

*(2)cd /team_code_transfuser.* 

*(3)(path to python) python pytorch_profiler_train.py (parameter)*

`python` `pytorch_profiler_train.py` `--batch_size` `10` `--logdir/home/niumo/transfuser-2022/logdir` `--root_dir` `/home/tangpeng/transfuser/data/ --parallel_training 0 --epoch 1`