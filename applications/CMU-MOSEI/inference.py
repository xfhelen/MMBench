import sys
import os
import torch
import torch.nn as nn
import argparse
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from models.unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from models.eval_scripts.complexity import all_in_one_train
from models.fusions.common_fusions import Concat

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--options', default="normal", type=str, help='choose the model part') 
args = parser.parse_args()
options = args.options

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MMDL(nn.Module):
    def __init__(self, encoders, fusion, head, has_padding=False):
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def encoder_part(self,inputs):
        head_out = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                head_out.append(self.encoders[i]([inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                head_out.append(self.encoders[i](inputs[i]))
        self.reps = head_out
        return head_out
    
    def fusion_part(self, encoder_out):
        if self.has_padding:
            if isinstance(encoder_out[0], torch.Tensor):
                fusion_out = self.fuse(encoder_out)
            else:
                fusion_out = self.fuse([i[0] for i in encoder_out])
        else:
            fusion_out = self.fuse(encoder_out)
        self.fuseout = fusion_out
        return fusion_out
    
    def header_part(self, fusion_out,encoder_out,inputs):
        if type(fusion_out) is tuple:
            fusion_out = fusion_out[0]
        if self.has_padding and not isinstance(encoder_out[0], torch.Tensor):
            return self.head([fusion_out, inputs[1][0]])
        return self.head(fusion_out)
    
    def forward(self, inputs):
        if options == 'normal':
            encoder_out = self.encoder_part(inputs)
            fusion_out = self.fusion_part(encoder_out)
            return self.header_part(fusion_out,encoder_out,inputs)
        elif options == 'encoder':
            return self.encoder_part(inputs)
        elif options == 'fusion':
            encoder_out = [torch.ones(1, 70).to(device),torch.ones(1, 200).to(device),torch.ones(1, 600).to(device)]
            return self.fusion_part(encoder_out)
        elif options == 'head':
            fusion_out = torch.ones(1, 870).to(device)
            encoder_out = [torch.ones(1, 70).to(device),torch.ones(1, 200).to(device),torch.ones(1, 600).to(device)]
            return self.header_part(fusion_out,encoder_out,inputs)
        else:
            print('Please choose the right options from normal, encoder, fusion, head')
            exit()


def train(encoders, fusion, head, valid_dataloader, total_epochs, is_packed=False,input_to_float=True,track_complexity=True):

    model = MMDL(encoders, fusion, head, has_padding=is_packed).to(device)

    def _trainprocess():
        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for _ in range(total_epochs):
            model.eval()
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/zhuxiaozhi/MMBench/applications/CMU-MOSEI/log'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    dataloader_iter = iter(valid_dataloader)
                    for _ in range(5):
                        try:
                            batch = next(dataloader_iter)
                            if is_packed:
                                _ = model([[_processinput(i).to(device) for i in batch[0]], batch[1]])
                            else:
                                _ = model([_processinput(i).to(device) for i in batch[:-1]])
                            prof.step()
                        except StopIteration:
                            break

    if track_complexity:
        all_in_one_train(_trainprocess, [model])
    else:
        _trainprocess()


if __name__ == '__main__':
    validdata = get_dataloader('datasets/affect/raw_data/')

    encoders = [GRU(713, 70, dropout=True, has_padding=True, batch_first=True).cuda(),
                GRU(74, 200, dropout=True, has_padding=True, batch_first=True).cuda(),
                GRU(300, 600, dropout=True, has_padding=True, batch_first=True).cuda()]
    head = MLP(870, 870, 1).cuda()
    fusion = Concat().cuda()

    train(encoders, fusion, head, validdata, 1, is_packed=True)
