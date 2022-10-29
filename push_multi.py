# From https://github.com/brentyi/multimodalfilter/blob/master/scripts/push_task/train_push.py
import torch.autograd.profiler as profiler
import torch.optim as optim
import torch.nn as nn
import torch
import fannypack
import datetime
import argparse
import sys
import os
import torch.profiler

sys.path.insert(0, os.getcwd())

from training_structures.Supervised_Learning import train, test  # noqa
from fusions.common_fusions import ConcatWithLinear  # noqa
from unimodals.gentle_push.head import GentlePushLateLSTM  # noqa
from unimodals.common_models import Sequential, Transpose, Reshape, MLP, Identity  # noqa
from datasets.gentle_push.data_loader import PushTask  # noqa
import unimodals.gentle_push.layers as layers  # noqa
from unimodals.gentle_push.head import Head  # noqa
from fusions.mult import MULTModel  # noqa
from fusions.common_fusions import TensorFusion  # noqa


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

    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        # with profiler.record_function("LINEAR PASS"):
        outs = []
        for i in range(len(inputs)):
            # with profiler.record_function("UNI_{}".format(i)):
            outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        # with profiler.record_function("FUSION"):
        out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        # with profiler.record_function("HEAD"):
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
    args = parser.parse_args()
    dataset_args = Task.get_dataset_args(args)
    fannypack.data.set_cache_path('datasets/gentle_push/cache')
    train_loader, val_loader, test_loader = Task.get_dataloader(
        16, batch_size=32, drop_last=True)
    # torch.save(train_loader, 'cache/train_push.data')
    # torch.save(val_loader, 'cache/val_push.data')
    # torch.save(test_loader, 'cache/test_push.data')


    # train_loader = torch.load('cache/train_push.data')
    # val_loader = torch.load('cache/val_push.data')
    # test_loader = torch.load('cache/test_push.data')

    # train_loader, val_loader, test_loader = torch.load('cache/push.data')
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MMDL(encoders, fusion, head, has_padding=False).to(device)
    # write string
    with open('./results/push_{}_arch.txt'.format(fusion_name), 'w', encoding='utf-8') as f:
            f.write(str(model))

    # output_names = ['prediction']
    # for j in val_loader:
    #     model.eval()
    #     torch.onnx.export(model, [i.float().to(device)
    #                               for i in j[:-1]], 
    #                       'results/pushuni_{}_agg.onnx'.format(fusion_name), export_params=False,
    #                       opset_version=11,
    #                       output_names=output_names)
    #     return 

    # hot
    for j in val_loader:
        model.eval()
        out = model([i.float().to(device) for i in j[:-1]])
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
                out = model([i.float().to(device) for i in j[:-1]])
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
    for f in ['lf', 'ef', 'mult', 'tf'][-2:-1]:
        print(f)
        run_multi(f)
