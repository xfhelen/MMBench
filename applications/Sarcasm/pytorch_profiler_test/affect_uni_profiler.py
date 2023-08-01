import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
from training_structures.unimodal import train, test  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import ConcatEarly  # noqa


if __name__ == '__main__':
# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# traindata, validdata, testdata = get_dataloader('/home/pliang/multibench/affect/pack/mosi/mosi_raw.pkl', robust_test=False)
    traindata, validdata, testdata = get_dataloader(
        'C:/Users/29296/Documents/Tencent Files/2929629852/FileRecv/sarcasm.pkl', robust_test=False, max_pad=True, data_type='sarcasm', max_seq_len=50)

    modality_num = 2

# mosi/mosei
    encoder = GRU(300, 600, dropout=True, has_padding=False,
              batch_first=True, last_only=True).cuda()
    head = MLP(600, 512, 1).cuda()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        train(encoder, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, lr=2e-3,
              weight_decay=0.01, criterion=torch.nn.L1Loss(), save_encoder='encoder.pt', save_head='head.pt',
              modalnum=modality_num)
    print(p.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=-1))

    # train(encoder, head, traindata, validdata, 1, task="regression", optimtype=torch.optim.AdamW, lr=2e-3,
    #       weight_decay=0.01, criterion=torch.nn.L1Loss(), save_encoder='encoder.pt', save_head='head.pt', modalnum=modality_num)

    print("Testing:")
    encoder = torch.load('encoder.pt').cuda()
    head = torch.load('head.pt')
    test(encoder, head, testdata, 'affect', criterion=torch.nn.L1Loss(),
         task="posneg-classification", modalnum=modality_num, no_robust=True)