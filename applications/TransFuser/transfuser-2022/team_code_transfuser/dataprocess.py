import numpy as np
import pandas as pd

data = pd.read_csv('trans-encoder.csv',header=None,lineterminator='\n')  # 获取到的csv文件

def turntolist(c):
    d=[]
    for i in range(0,len(c)):
        d.append(c[0][i])
    f=[]
    for i in range(0,len(d)):
        if type(d[i])==str:
            split=d[i].split()
            f.append(split)
                    
    return f

data = turntolist(data)


def count(l):
    result=[0,0,0,0,0]
    count=0
    for i in range(0,len(l)):
        if l[i][0]=='dram__throughput.avg.pct_of_peak_sustained_elapsed':
            result[0]=result[0]+float(l[i][2])
            count=count+1
        if l[i][0]=='sm__warps_active.avg.pct_of_peak_sustained_active':
            result[1]=result[1]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__inst_executed.avg.per_cycle_active':
            result[2]=result[2]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct':
            result[3]=result[3]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct':
            result[4]=result[4]+float(l[i][2])
            count=count+1
    count=count/5
    print(count)
    for i in range(0,5):
        result[i]=result[i]/count
    return result

def count2(l):
    result=[0,0,0,0,0,0,0]
    count=0
    for i in range(0,len(l)):
        if l[i][0]=='smsp__warp_issue_stalled_imc_miss_per_warp_active.pct':
            result[0]=result[0]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct':
            result[1]=result[1]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__warp_issue_stalled_wait_per_warp_active.pct':
            result[1]=result[1]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__warp_issue_stalled_no_instruction_per_warp_active.pct':
            result[2]=result[2]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct':
            result[3]=result[3]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__warp_issue_stalled_not_selected_per_warp_active.pct':
            result[4]=result[4]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct':
            result[5]=result[5]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct':
            result[5]=result[5]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__warp_issue_stalled_barrier_per_warp_active.pct':
            result[6]=result[6]+float(l[i][2])
            count=count+1
        if l[i][0]=='smsp__warp_issue_stalled_membar_per_warp_active.pct':
            result[6]=result[6]+float(l[i][2])
            count=count+1
    count=count/10
    print(count)
    for i in range(0,7):
        result[i]=result[i]/count
    return result

def count3(l):
    result=[0,0,0,0]
    count=0 
    for i in range(0,len(l)):
        if l[i][0]=='gpu__time_active.avg':
            count=count+1
            if float(l[i][2])< 10:
                result[0]=result[0]+1
            if float(l[i][2])<50 and float(l[i][2])>=10:
                result[1]=result[1]+1
            if float(l[i][2])<100 and float(l[i][2])>=50:
                result[2]=result[2]+1
            if float(l[i][2])>=100:
                result[3]=result[3]+1
    print(count)
    return result

def count2nvprof(l):
    result=[0,0,0,0,0,0,0]
    count=0
    for i in range(0,len(l)):
        if l[i][1]=='stall_constant_memory_dependency':
            result[0]=result[0]+float(l[i][9].replace('%','0'))
            count=count+1
        if l[i][1]=='stall_exec_dependency':
            result[1]=result[1]+float(l[i][9].replace('%','0'))
            count=count+1
        if l[i][1]=='stall_inst_fetch':
            result[2]=result[2]+float(l[i][9].replace('%','0'))
            count=count+1
        if l[i][1]=='stall_memory_dependency':
            result[3]=result[3]+float(l[i][9].replace('%','0'))
            count=count+1
        if l[i][1]=='stall_not_selected':
            result[4]=result[4]+float(l[i][9].replace('%','0'))
            count=count+1
        if l[i][1]=='stall_pipe_busy':
            result[5]=result[5]+float(l[i][9].replace('%','0'))
            count=count+1
        if l[i][1]=='stall_sync':
            result[6]=result[6]+float(l[i][8].replace('%','0'))
            count=count+1   
    count=count/7
    print(count)
    for i in range(0,7):
        result[i]=result[i]/count
    return result

def countnvprof(l):
    result=[0,0,0,0,0]
    count=0
    for i in range(0,len(l)):
        if l[i][1]=='dram_utilization':
            result[0]=result[0]+float(l[i][10][-3:-1].replace('(',' '))
            count=count+1
        if l[i][1]=='achieved_occupancy':
            result[1]=result[1]+float(l[i][6])
            count=count+1
        if l[i][1]=='ipc':
            result[2]=result[2]+float(l[i][6])
            count=count+1
        if l[i][1]=='gld_efficiency':
            result[3]=result[3]+float(l[i][8].replace('%','0'))
            count=count+1
        if l[i][1]=='gst_efficiency':
            result[4]=result[4]+float(l[i][8].replace('%','0'))
            count=count+1 
    count=count/5
    print(count)
    for i in range(0,5):
        result[i]=result[i]/count
    return result

result=count(data)
print(result)