import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import math
from typing import Tuple
from models.conmamba import ConmambaBlock


class Linear(nn.Module):
    def __init__(self,
                 noise_length: int,
                 label_length: int
                 ) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features=noise_length, out_features=label_length),
            nn.ReLU()
        )
    def forward(self, input: Tensor) -> Tensor:
        output = torch.transpose(input, 1, 2)
        output = self.linear(output)
        return output.transpose(1,2)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Conv2dUpampling(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_dropout_p: int,
                 kernel_size=3
                 ) -> None:
        super(Conv2dUpampling, self).__init__()
        padding=calc_same_padding(kernel_size)
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(p=conv_dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        outputs = self.dropout(outputs)
        return outputs


class Encoder(nn.Module):
    def __init__(self,
                 dim: int = None
                 ):
        super(Encoder, self).__init__()

        self.upsampling = Conv2dUpampling(
            in_channels=1,
            out_channels=dim//4,
            conv_dropout_p=0.1
        )

        self.conmamba = ConmambaBlock(
            dim=dim,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.1,
            ff_dropout=0.1,
            conv_dropout=0.1
        )

    def forward(self, x):
        x = self.upsampling(x)
        x = self.conmamba(x)
        return x


class RNNBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 lstm_hidden_dim: int,
                 rnn_dropout_p=0.1,
                 ) -> None:
        super(RNNBlock, self).__init__()
        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_dim, num_layers=2, bidirectional=False,
                           batch_first=True)
        self.linear = nn.Linear(in_features=lstm_hidden_dim, out_features=4)
        self.dropout = nn.Dropout(rnn_dropout_p)


    def forward(self, input: Tensor) -> Tensor:
        output, _ = self.rnn(input)
        output = self.linear(output)
        output = self.dropout(output)

        return F.softplus(output)


def ds_fusion(evidence: torch.Tensor) -> torch.Tensor:
    fused_evidence = torch.mean(evidence, dim=1)
    return fused_evidence


class Model(nn.Module):
    def __init__(self,
                 encoder,
                 dim: int,
                 noise_length: int,
                 label_length: int,
                 ) -> None:
        super(Model, self).__init__()
        self.encoder = encoder
        self.length_adapter = nn.Linear(noise_length, label_length)
        self.rnnblock = RNNBlock(in_channels=dim,
                                 lstm_hidden_dim=256,
                                 rnn_dropout_p=0.1)

    def forward(self, input: Tensor) -> Tensor:
        B, N, L, D = input.shape
        input = input.view(B * N, L, D)
        encoded = self.encoder(input)
        encoded = encoded.permute(0, 2, 1)
        encoded = self.length_adapter(encoded)
        encoded = encoded.permute(0, 2, 1)
        evidence = self.rnnblock(encoded)
        evidence = evidence.view(B, N, evidence.size(1), evidence.size(2))
        fused_evidence = ds_fusion(evidence)  
        return fused_evidence





