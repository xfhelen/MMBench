"""Implements common fusion patterns."""

import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self, concat_dim=1):
        """Initialize Concat Module.
        
        :param concat_dim: The concatentation dimension for the modalities.
        """
        super(Concat, self).__init__()
        self.concat_dim = concat_dim

    def forward(self, modalities):
        """
        Forward Pass of Concat.
        
        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=self.concat_dim)



# Stacking modalities
class Stack(nn.Module):
    """Stacks modalities together on dimension 1."""

    def __init__(self):
        """Initialize Stack Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of Stack.
        
        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.stack(flattened, dim=2)


class ConcatWithLinear(nn.Module):
    """Concatenates input and applies a linear layer."""

    def __init__(self, input_dim, output_dim, concat_dim=1):
        """Initialize ConcatWithLinear Module.
        
        :param input_dim: The input dimension for the linear layer
        :param output_dim: The output dimension for the linear layer
        :concat_dim: The concatentation dimension for the modalities.
        """
        super(ConcatWithLinear, self).__init__()
        self.concat_dim = concat_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, modalities):
        """
        Forward Pass of Stack.
        
        :param modalities: An iterable of modalities to combine
        """
        return self.fc(torch.cat(modalities, dim=self.concat_dim))


class FeatureModulator(nn.Module):
    """FiLM 用于生成gamma和beta参数的网络"""
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def create_film_layer(image_dim=256, text_dim=512, hidden_dim=512):
    """
    创建FiLM层
    :param image_dim: 图像特征维度
    :param text_dim: 文本特征维度
    :param hidden_dim: 隐藏层维度
    :return: 配置好的FiLM层
    """
    # 创建gamma生成网络
    gamma_net = FeatureModulator(
        input_dim=text_dim,    
        output_dim=image_dim,   
        hidden_dim=hidden_dim
    )
    
    # 创建beta生成网络
    beta_net = FeatureModulator(
        input_dim=text_dim,    
        output_dim=image_dim,  
        hidden_dim=hidden_dim
    )
    
    # 创建FiLM层
    film_layer = FiLM(
        gamma_generation_network=gamma_net,
        beta_generation_network=beta_net,
        base_modal=0,              # 图像特征是第一个模态
        gamma_generate_modal=1,    # 文本特征是第二个模态
        beta_generate_modal=1      # 文本特征是第二个模态
    )
    
    return film_layer


def create_film_layer_v2(image_dim=256, text_dim=512, audio_dim=128, hidden_dim=512):
    """
    创建使用不同模态生成gamma和beta的FiLM层
    
    :param image_dim: 图像特征维度
    :param text_dim: 文本特征维度
    :param audio_dim: 音频特征维度
    :return: 配置好的FiLM层
    """
    # gamma使用文本特征生成
    gamma_net = FeatureModulator(
        input_dim=text_dim,   
        output_dim=image_dim,  
        hidden_dim=hidden_dim
    )
    
    # beta使用音频特征生成
    beta_net = FeatureModulator(
        input_dim=audio_dim,  
        output_dim=image_dim,
        hidden_dim=hidden_dim
    )
    
    # 创建FiLM层
    film_layer = FiLM(
        gamma_generation_network=gamma_net,
        beta_generation_network=beta_net,
        base_modal=0,              # 图像特征是第一个模态
        gamma_generate_modal=1,    # 文本特征是第二个模态
        beta_generate_modal=2      # 音频特征是第三个模态
    )
    
    return film_layer

class FiLM(nn.Module):
    """Implements FiLM - Feature-Wise Affine Transformations of the Input.
    
    See https://arxiv.org/pdf/1709.07871.pdf for more details.
    """

    def __init__(self, gamma_generation_network, beta_generation_network, base_modal=0, gamma_generate_modal=1, beta_generate_modal=1):
        """Initialize FiLM layer.
        
        :param gamma_generation_network: Network which generates gamma_parameters from gamma_generation_modal data.
        :param beta_generation_network: Network which generates beta_parameters from beta_generation_modal data.
        :param base_modal: Modality to apply affine transformation to.
        :param gamma_generate_modal: Modality to generate gamma portion of affine transformation from.
        :param beta_generate_modal: Modality to generate beta portion of affine transformation from.
        """
        super(FiLM, self).__init__()
        self.g_net = gamma_generation_network
        self.b_net = beta_generation_network
        self.base_modal = base_modal
        self.ggen_modal = gamma_generate_modal
        self.bgen_modal = beta_generate_modal

    def forward(self, modalities):
        """
        Forward Pass of FiLM.
        
        :param modalities: An iterable of modalities to combine. 
        """
        for i in range(len(modalities)):
            modalities[i] = modalities[i].float()
        gamma = self.g_net(modalities[self.ggen_modal])
        beta = self.b_net(modalities[self.bgen_modal])
        return gamma * modalities[self.base_modal] + beta



class MultiplicativeInteractions3Modal(nn.Module):
    """Implements 3-Way Modal Multiplicative Interactions."""
    
    def __init__(self, input_dims, output_dim, output, flatten=False, clip=None, grad_clip=None, flip=False):
        """Initialize MultiplicativeInteractions3Modal object.

        :param input_dims: list or tuple of 3 integers indicating sizes of input
        :param output_dim: size of outputs
        """
        super(MultiplicativeInteractions3Modal, self).__init__()
            
        self.a = MultiplicativeInteractions2Modal([input_dims[0], input_dims[1]],
                                                  [input_dims[2], output_dim],  output, flatten, clip, grad_clip, flip)
        self.b = MultiplicativeInteractions2Modal([input_dims[0], input_dims[1]],
                                                  output_dim, output, flatten, clip, grad_clip, flip)

    def forward(self, modalities):
        """
        Forward Pass of MultiplicativeInteractions3Modal.
        
        :param modalities: An iterable of modalities to combine. 
        """
        # if self.task == 'affect':
        #     return torch.einsum('bm, bmp -> bp', modalities[2], self.a(modalities[0:2])) + self.b(modalities[0:2])
        return torch.matmul(modalities[2], self.a(modalities[0:2])) + self.b(modalities[0:2])


class MultiplicativeInteractions2Modal(nn.Module):
    """Implements 2-way Modal Multiplicative Interactions."""
    
    def __init__(self, input_dims, output_dim, output, flatten=False, clip=None, grad_clip=None, flip=False):
        """
        :param input_dims: list or tuple of 2 integers indicating input dimensions of the 2 modalities
        :param output_dim: output dimension
        :param output: type of MI, options from 'matrix3D','matrix','vector','scalar'
        :param flatten: whether we need to flatten the input modalities
        :param clip: clip parameter values, None if no clip
        :param grad_clip: clip grad values, None if no clip
        :param flip: whether to swap the two input modalities in forward function or not
        
        """
        super(MultiplicativeInteractions2Modal, self).__init__()
        self.input_dims = input_dims
        self.clip = clip
        self.output_dim = output_dim
        self.output = output
        self.flatten = flatten
        if output == 'matrix3D':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(
                input_dims[0], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(
                input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.b)

        # most general Hypernetworks as Multiplicative Interactions.
        elif output == 'matrix':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0], output_dim))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(input_dims[1], output_dim))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim))
            nn.init.normal_(self.b)
        # Diagonal Forms and Gating Mechanisms.
        elif output == 'vector':
            self.W = nn.Parameter(torch.Tensor(input_dims[0], input_dims[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(
                self.input_dims[0], self.input_dims[1]))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.b)
        # Scales and Biases.
        elif output == 'scalar':
            self.W = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.b)
        self.flip = flip
        if grad_clip is not None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(
                    grad, grad_clip[0], grad_clip[1]))

    def _repeatHorizontally(self, tensor, dim):
        return tensor.repeat(dim).view(dim, -1).transpose(0, 1)

    def forward(self, modalities):
        """
        Forward Pass of MultiplicativeInteractions2Modal.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]
        elif len(modalities) > 2:
            assert False
        m1 = modalities[0].float()
        m2 = modalities[1].float()
        if self.flip:
            m1 = modalities[1]
            m2 = modalities[0]

        if self.flatten:
            m1 = torch.flatten(m1, start_dim=1)
            m2 = torch.flatten(m2, start_dim=1)
        if self.clip is not None:
            m1 = torch.clip(m1, self.clip[0], self.clip[1])
            m2 = torch.clip(m2, self.clip[0], self.clip[1])

        if self.output == 'matrix3D':
            Wprime = torch.einsum('bn, nmpq -> bmpq', m1,
                                  self.W) + self.V  # bmpq
            bprime = torch.einsum('bn, npq -> bpq', m1,
                                  self.U) + self.b    # bpq
            output = torch.einsum('bm, bmpq -> bpq', m2,
                                  Wprime) + bprime   # bpq

        # Hypernetworks as Multiplicative Interactions.
        elif self.output == 'matrix':
            Wprime = torch.einsum('bn, nmd -> bmd', m1,
                                  self.W) + self.V      # bmd
            bprime = torch.matmul(m1, self.U) + self.b      # bmd
            output = torch.einsum('bm, bmd -> bd', m2,
                                  Wprime) + bprime             # bmd

        # Diagonal Forms and Gating Mechanisms.
        elif self.output == 'vector':
            Wprime = torch.matmul(m1, self.W) + self.V      # bm
            bprime = torch.matmul(m1, self.U) + self.b      # b
            output = Wprime*m2 + bprime             # bm

        # Scales and Biases.
        elif self.output == 'scalar':
            Wprime = torch.matmul(m1, self.W.unsqueeze(1)).squeeze(1) + self.V
            bprime = torch.matmul(m1, self.U.unsqueeze(1)).squeeze(1) + self.b
            output = self._repeatHorizontally(
                Wprime, self.input_dims[1]) * m2 + self._repeatHorizontally(bprime, self.input_dims[1])
        return output


class TensorFusion(nn.Module):
    """
    Implementation of TensorFusion Networks.
    
    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """
    def __init__(self):
        """Instantiates TensorFusion Network Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat((Variable(torch.ones(
            *nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0), dim=-1)
        for mod in modalities[1:]:
            mod = torch.cat((Variable(torch.ones(
                *nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod), dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m


class LowRankTensorFusion(nn.Module):
    """
    Implementation of Low-Rank Tensor Fusion.
    
    See https://github.com/Justin1904/Low-rank-Multimodal-Fusion for more information.
    """

    def __init__(self, input_dims, output_dim, rank, flatten=True):
        """
        Initialize LowRankTensorFusion object.
        
        :param input_dims: list or tuple of integers indicating input dimensions of the modalities
        :param output_dim: output dimension
        :param rank: a hyperparameter of LRTF. See link above for details
        :param flatten: Boolean to dictate if output should be flattened or not. Default: True
        
        """
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        # low-rank factors
        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(
                self.rank, input_dim+1, self.output_dim).float(), requires_grad=False).to(device)
            nn.init.xavier_normal_(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank).float(), requires_grad=False).to(device)
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim).float(), requires_grad=False).to(device)
        # init the fusion weights
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):
        """
        Forward Pass of Low-Rank TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        batch_size = modalities[0].shape[0]
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            modality = modality.float()
            ones = Variable(torch.ones(batch_size, 1).type(
                modality.dtype), requires_grad=False).to(device)
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


class NLgate(torch.nn.Module):
    """
    Implements of Non-Local Gate-based Fusion.

    
    See section F4 of https://arxiv.org/pdf/1905.12681.pdf for details
    """
    
    def __init__(self, thw_dim, c_dim, tf_dim, q_linear=None, k_linear=None, v_linear=None):
        """
        q_linear, k_linear, v_linear are none if no linear layer applied before q,k,v.
        
        Otherwise, a tuple of (indim,outdim) is required for each of these 3 arguments.
        
        :param thw_dim: See paper
        :param c_dim: See paper
        :param tf_dim: See paper
        :param q_linear: See paper
        :param k_linear: See paper
        :param v_linear: See paper
        """
        super(NLgate, self).__init__()
        self.qli = None
        if q_linear is not None:
            self.qli = nn.Linear(q_linear[0], q_linear[1])
        self.kli = None
        if k_linear is not None:
            self.kli = nn.Linear(k_linear[0], k_linear[1])
        self.vli = None
        if v_linear is not None:
            self.vli = nn.Linear(v_linear[0], v_linear[1])
        self.thw_dim = thw_dim
        self.c_dim = c_dim
        self.tf_dim = tf_dim
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Apply Low-Rank TensorFusion to input.
        
        :param x: An iterable of modalities to combine. 
        """
        q = x[0]
        k = x[1]
        v = x[1]
        if self.qli is None:
            qin = q.view(-1, self.thw_dim, self.c_dim)
        else:
            qin = self.qli(q).view(-1, self.thw_dim, self.c_dim)
        if self.kli is None:
            kin = k.view(-1, self.c_dim, self.tf_dim)
        else:
            kin = self.kli(k).view(-1, self.c_dim, self.tf_dim)
        if self.vli is None:
            vin = v.view(-1, self.tf_dim, self.c_dim)
        else:
            vin = self.vli(v).view(-1, self.tf_dim, self.c_dim)
        matmulled = torch.matmul(qin, kin)
        finalout = torch.matmul(self.softmax(matmulled), vin)
        return torch.flatten(qin + finalout, 1)


class AttentionFusion(nn.Module):
    def __init__(self, input_dims, output_dim):
        super(AttentionFusion, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for input_dim in input_dims])
        self.attention = nn.Linear(output_dim * len(input_dims), len(input_dims))

    def forward(self, modalities):
        transformed = [linear(modality.float()) for linear, modality in zip(self.linears, modalities)]
        concatenated = torch.cat(transformed, dim=1)
        attention_weights = torch.softmax(self.attention(concatenated), dim=1)
        weighted_sum = sum(w * t for w, t in zip(attention_weights.split(1, dim=1), transformed))
        return weighted_sum


class RNNFusion(nn.Module):
    def __init__(self,output_dim):
        super(RNNFusion, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=output_dim, num_layers=2, batch_first=True)

    def forward(self, modalities):
        # 将每个模态展平并转换为序列
        flattened = []
        for modality in modalities:
            modality = modality.float()
            flat = torch.flatten(modality, start_dim=1).unsqueeze(2)
            flattened.append(flat)
            
        # 将所有模态拼接为序列
        sequence = torch.cat(flattened, dim=1)
        
        # 通过GRU处理序列
        output, hidden = self.gru(sequence)
        
        # 返回最后一个时间步的输出
        return output[:,-1,:]