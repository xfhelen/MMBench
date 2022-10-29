#simple-late-fusion
# encoder
sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_simple_late_fusion.py >avmnist-slfs-encoder.csv

#fusion
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_simple_late_fusion.py >avmnist-slfs-fusion.csv

#head
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_simple_late_fusion.py >avmnist-slfs-head.csv


#cca
# encoder
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_cca.py >avmnist-cca-encoder.csv

#fusion
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_cca.py >avmnist-cca-fusion.csv

#head
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_cca.py >avmnist-cca-head.csv


#multi-interac-matrix
# encoder
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_multi_interac_matrix.py >avmnist-multi-encoder.csv

#fusion
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_multi_interac_matrix.py >avmnist-multi-fusion.csv

#head
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_multi_interac_matrix.py >avmnist-multi-head.csv


#low rank tensor
# encoder
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_low_rank_tensor.py >avmnist-tensor-encoder.csv

#fusion
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_low_rank_tensor.py >avmnist-tensor-fusion.csv

#head
#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active,pcie__read_bytes.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,gpu__time_active.avg /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_low_rank_tensor.py >avmnist-tensor-head.csv





#sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,lts__t_sector_op_read_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum.per_second,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum.per_second,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_unimodal_0.py >avmnist-uni-v2.csv


#²âÀ×´ïÍ¼
sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__inst_executed.avg.per_cycle_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_unimodal_0.py >avmnist-uni-radar.csv


#²âstall
sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/ncu --metrics smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_sleeping_per_warp_active.pct,smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct /home/xucheng/.conda/envs/python3.7/bin/python3  /home/xucheng/MultiBench/examples/multimedia/avmnist_simple_late_fusion.py >slfs-fusion-stall.csv


