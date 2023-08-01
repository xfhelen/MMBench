import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from unimodals.common_models import Transformer, MLP, Sequential, Identity # noqa
from models.eval_scripts.complexity import  all_in_one_train, all_in_one_test
from training_structures.Supervised_Learning import train, test # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import ConcatEarly # noqa

if __name__ == '__main__':

# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_senti_data.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
    traindata, validdata, testdata = get_dataloader(
    'C:/Users/29296/Documents/Tencent Files/2929629852/FileRecv/sarcasm.pkl', robust_test=False,data_type='sarcasm')

# mosi/mosei
# encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
# head = Sequential(Transformer(409, 300).cuda(), MLP(300, 128, 1)).cuda()

# humor/sarcasm
    encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
    head = Sequential(Transformer(752, 300).cuda(),MLP(300, 128, 1)).cuda()


    fusion = ConcatEarly().cuda()

#可以使用
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
    ) as p:
        train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, is_packed=True, early_stop=True,lr=1e-4, save='mosi_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())
    print(p.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=-1))


    #train(encoders, fusion, head, traindata, validdata, 2, task="regression", optimtype=torch.optim.AdamW, is_packed=True, early_stop=True,lr=1e-4, save='mosi_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

    print("Testing:")
    model = torch.load('mosi_ef_best.pt').cuda()
    test(model, testdata, 'affect', is_packed=True,
         criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)