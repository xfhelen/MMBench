import nvitop
import torch
import sys
import os
import subprocess
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append("C:\\Users\\29296\\Desktop\\MMBench-main\\models")
sys.path.append("C:\\Users\\29296\\Desktop\\MMBench-main\\datasets")
# print(sys.path)
# print("123")
# print("123")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models.unimodals.common_models import GRU, MLP, Sequential, Identity,GRUWithLinear  # noqa
from models.examples.affect.Supervised_Learning_2 import train, test  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from models.fusions.common_fusions import ConcatEarly  # noqa
#from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from memory_profiler import memory_usage, profile  # noqa
from models.eval_scripts.complexity import  all_in_one_train, all_in_one_test
import psutil

if __name__ == '__main__':

    print(torch.cuda.is_available())

    print(torch.cuda.device_count())

    #subprocess.Popen(["start", "cmd", "/k", "C:\\Anaconda3\\envs\\pytorch\\Scripts\\nvitop.exe"])

    traindata, validdata, testdata = get_dataloader('C:/Users/29296/Documents/Tencent Files/2929629852/FileRecv/sarcasm.pkl', robust_test=False, max_pad=True,  data_type='sarcasm', max_seq_len=40)

    encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
    # encoders = [GRUWithLinear(371, 512, 32, dropout=True, has_padding=True).cuda(), \
    #             GRUWithLinear(81, 256, 32, dropout=True, has_padding=True).cuda(), \
    #             GRUWithLinear(300, 600, 128, dropout=True, has_padding=True).cuda()]
    head = Sequential(GRU(752, 1128, dropout=True, has_padding=False, batch_first=True, last_only=True), MLP(1128, 512, 1)).cuda()

    fusion = ConcatEarly().cuda()

    #allmodules = [encoders[0], encoders[1], encoders[2], head, fusion]
    @profile
    def train_process():
        train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
            is_packed=False, lr=1e-3, save='sarcasm_temp.pt', weight_decay=0.01, objective=torch.nn.L1Loss())


    #all_in_one_train(train_process, allmodules)
    print("begin")
    train_process()

    print("Testing:")
    model = torch.load('sarcasm_temp.pt').cuda()
    test(model, testdata, 'affect', is_packed=False,
        criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
