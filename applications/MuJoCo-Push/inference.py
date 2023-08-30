import torch.optim as optim
import torch.nn as nn
from torch import nn
import torch
import fannypack
import datetime
import argparse
import sys
import os
import torch.profiler
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from models.training_structures.Supervised_Learning import train, test  # noqa
from models.fusions.common_fusions import ConcatWithLinear  # noqa
from models.unimodals.gentle_push.head import GentlePushLateLSTM  # noqa
from models.unimodals.common_models import Sequential, Transpose, Reshape, MLP, Identity  # noqa
from datasets.gentle_push.data_loader import PushTask  # noqa
import models.unimodals.gentle_push.layers as layers  # noqa
from models.unimodals.gentle_push.head import Head  # noqa
from models.fusions.mult import MULTModel  # noqa
from models.fusions.common_fusions import TensorFusion  # noqa

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MMDL(nn.Module):
    """Implements MMDL classifier."""
    
    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs, options):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if options == 'normal':
            # with profiler.record_function("LINEAR PASS"):
            outs = []
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
                # print(outs[i].shape)
            self.reps = outs
            # with profiler.record_function("FUSION"):
            out = self.fuse(outs)
            self.fuseout = out
            if type(out) is tuple:
                out = out[0]
            # print(out.shape)
            # with profiler.record_function("HEAD"):
            return self.head(out)

        elif options == 'encoder':
            outs = []
            for i in range(len(inputs)):
                # with profiler.record_function("UNI_{}".format(i)):
                outs.append(self.encoders[i](inputs[i]))
            self.reps = outs

        elif options == 'fusion':
            outs = []
            outs.append(torch.zeros([1,16,256]).to(device))
            outs.append(torch.zeros([1,16,256]).to(device))
            outs.append(torch.zeros([1,16,256]).to(device))
            outs.append(torch.zeros([1,16,256]).to(device))
            out = self.fuse(outs)
            return out

        elif options == 'head':
            # with profiler.record_function("HEAD"):
            out = torch.zeros([1,  16, 2])
            return self.head(out)


class HyperParams(MULTModel.DefaultHyperParams):
    num_heads = 4
    embed_dim = 64
    output_dim = 2
    all_steps = True


def run_multi(fusion_name):
    # Parse args
    Task = PushTask
    parser = argparse.ArgumentParser()
    Task.add_dataset_arguments(parser)
    parser.add_argument('--options', default="normal", type=str, help='choose the model part')
    args = parser.parse_args()
    dataset_args = Task.get_dataset_args(args)
    fannypack.data.set_cache_path('datasets/gentle_push/cache')
    parser = argparse.ArgumentParser()
    options = args.options

    val_loader= Task.get_dataloader(
        16, batch_size=1, drop_last=True)

    if fusion_name == 'lf':
        encoders = [
            Sequential(Transpose(0, 1), layers.observation_pos_layers(
                64), GentlePushLateLSTM(64, 256), Transpose(0, 1)),
            Sequential(Transpose(0, 1), layers.observation_sensors_layers(
                64), GentlePushLateLSTM(64, 256), Transpose(0, 1)),
            Sequential(Transpose(0, 1), Reshape([-1, 1, 32, 32]), layers.observation_image_layers(
                64), Reshape([16, -1, 64]), GentlePushLateLSTM(64, 256), Transpose(0, 1)),
            Sequential(Transpose(0, 1), layers.control_layers(64),
                    GentlePushLateLSTM(64, 256), Transpose(0, 1)),
        ]
        fusion = ConcatWithLinear(256 * 4, 2, concat_dim=2)
        head = Identity()
    elif fusion_name == 'ef':
        encoders = [
            Sequential(Transpose(0, 1), layers.observation_pos_layers(64)),
            Sequential(Transpose(0, 1), layers.observation_sensors_layers(64)),
            Sequential(Transpose(0, 1), Reshape(
                [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64])),
            Sequential(Transpose(0, 1), layers.control_layers(64)),
        ]
        fusion = ConcatWithLinear(64 * 4, 64, concat_dim=2)
        head = Sequential(Head(), Transpose(0, 1))
    elif fusion_name == 'mult':
        encoders = [
            Sequential(Transpose(0, 1), layers.observation_pos_layers(
                64), Transpose(0, 1)),
            Sequential(Transpose(0, 1), layers.observation_sensors_layers(
                64), Transpose(0, 1)),
            Sequential(Transpose(0, 1), Reshape(
                [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
            Sequential(Transpose(0, 1), layers.control_layers(64), Transpose(0, 1)),
        ]
        fusion = MULTModel(4, [64, 64, 64, 64], HyperParams)
        head = Identity()
    elif fusion_name == 'tf':
        encoders = [
            Sequential(Transpose(0, 1), layers.observation_pos_layers(
                8), Transpose(0, 1)),
            Sequential(Transpose(0, 1), layers.observation_sensors_layers(
                8), Transpose(0, 1)),
            Sequential(Transpose(0, 1), Reshape(
                [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
            Sequential(Transpose(0, 1), layers.control_layers(
                16), Transpose(0, 1)),
        ]
        fusion = TensorFusion()
        head = MLP((8 + 1) * (8 + 1) * (64 + 1) * (16 + 1), 256, 2)
    optimtype = optim.Adam
    loss_state = nn.MSELoss()


    model = MMDL(encoders, fusion, head, has_padding=False).to(device)

    for j in val_loader:
        model.eval()
        out = model([i.float().to(device) for i in j[:-1]], options)
        break

    options = "normal"
    print('start inference')
    with torch.no_grad():
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './log/push_{}_agg'.format(fusion_name)),
            record_shapes=True,
            with_stack=True)
        with prof as p:
            for j in val_loader:
                model.eval()
                out = model([i.float().to(device) for i in j[:-1]], options)
                p.step()
        return
        # CUDA
        prof.export_stacks("results/profiler_stacks.txt",
                           "self_cuda_time_total")
        os.system(
            'cd FlameGraph;./flamegraph.pl --title "CUDA time" --countname "us." ../results/profiler_stacks.txt > ../results/gpu_perf_viz_{}.svg'.format(fusion_name))
        # CPU
        prof.export_stacks("results/profiler_stacks.txt",
                           "self_cpu_time_total")
        os.system(
            'cd FlameGraph;./flamegraph.pl --title "CPU time" --countname "us." ../results/profiler_stacks.txt > ../results/cpu_perf_viz_{}.svg'.format(fusion_name))
        # mem
        with open('./results/push_{}_agg.txt'.format(fusion_name), 'w', encoding='utf-8') as f:
            f.write(prof.key_averages().table(sort_by="self_cpu_memory_usage"))


if __name__ == '__main__':
    for f in ['lf', 'ef', 'mult', 'tf'][0:1]:
        run_multi(f)
