import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
from models.examples.Sarcasm.Supervised_Learning_2 import train, test  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import ConcatEarly  # noqa
#from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from memory_profiler import memory_usage # noqa
from models.eval_scripts.complexity import  all_in_one_train, all_in_one_test
import torch
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    traindata, validdata, testdata = get_dataloader(
        'C:/Users/29296/Documents/Tencent Files/2929629852/FileRecv/sarcasm.pkl', robust_test=False, max_pad=True,
        data_type='sarcasm', max_seq_len=40)

    encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
    head = Sequential(GRU(752, 1128, dropout=True, has_padding=False, batch_first=True, last_only=True),
                      MLP(1128, 512, 1)).cuda()
    fusion = ConcatEarly().cuda()
    allmodules = [encoders[0], encoders[1], encoders[2], head, fusion]


    def train_process():

            train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
                  is_packed=False, lr=1e-3, save='sarcasm_temp.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

    #
    # with torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA,
    #         ]
    # ) as p:
    #     train_process()
    # print(p.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=-1))

    #
    # with torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA,
    #             # torch.profiler.ProfilerActivity.,  # Corrected attribute name
    #         ],
    #         schedule=torch.profiler.schedule(
    #             wait=1,
    #             warmup=1,
    #             active=3,
    #             repeat=2
    #         ),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #             "C:/Users/29296/Desktop/实验室/MMBench-main/log/train_process"
    #         ),
    #         record_shapes=True,
    #         with_stack=True
    # ) as prof:
    #     all_in_one_train(train_process, allmodules)
    # print("123")
    # print(prof.key_averages().table())
    #
    # print("completed")

    # Non-default profiler schedule allows user to turn profiler on and off
    # on different iterations of the training loop;
    # trace_handler is called every time a new trace becomes available
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))
        # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")


    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2,
                repeat=1),
            #on_trace_ready=trace_handler
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
    ) as p:
        for iter in range(1):
            train_process()
            # send a signal to the profiler that the next iteration has started
            p.step()

    print(p.key_averages().table())

