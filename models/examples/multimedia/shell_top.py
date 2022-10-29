import os
import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model_name', help='name of model')
    parser.add_argument('--gpu', type=int, default=0, help='name of model')

    args = parser.parse_args()
    return args
args= args_parser()

### mmimdb_simple_early_fusion
lrs=[0.1*4e-2,0.5*4e-2,4e-2,5*4e-2,10*4e-2]
weight_decays=[0.001,0.005,0.01,0.03,0.05]
random_seeds=[8,17,23]
early_exit_steps=[10,15,20]
###mmimdb_low_rank_tensor
lr_base=8e-3
weight_decay_base=0.01


lrs=[0.1*lr_base,0.5*lr_base,lr_base,lr_base*5,lr_base*10]
weight_decays=[0.01*weight_decay_base,weight_decay_base*0.5,weight_decay_base,weight_decay_base*3,weight_decay_base*5]
random_seeds=[8,17,23]
early_exit_steps=[10,15,20]
index_count=0
for i,lr in enumerate(lrs):
    for j,weight_decay in enumerate(weight_decays):
        for k,random_seed in enumerate(random_seeds):
            for l,early_exit_step in enumerate(early_exit_steps):
                command='python3 mmimdb_simple_late_fusion.py --model_name {} --gpu {} --model_index {} --lr {} --weight_decay {}\
                    --random_seed {} --early_exit_step {} '.format(args.model_name,args.gpu,index_count,lr,weight_decay,random_seed,early_exit_step)
                file_log_all = open("log_all_acc.txt",'a',encoding='utf-8')
                file_log_all.write(' model_name:{}  gpu:{}  model_index:{}  lr:{}  weight_decay:{}\
                     random_seed:{}  early_exit_step:{} '.format(args.model_name,args.gpu,index_count,lr,weight_decay,random_seed,early_exit_step))
                file_log_all.flush()
                file_log_all.close()
                os.system(command)
                index_count+=1