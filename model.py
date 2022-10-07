import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim


class VGG(nn.Module):
    """
    VGG builder
    """

    def __init__(self, arch: object) -> object:
        super(VGG, self).__init__()
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(64, arch[2])
        self.conv3_512a = self.__make_layer(16, arch[3])
        self.conv3_512b = self.__make_layer(4, arch[4])
        self.fc1 = nn.Linear(540, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 5)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())  # [1, 2, 8, 3, 300, 480]
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4], x.shape[5]).squeeze(0)
        # output = torch.zeros([x.shape[1], x.shape[2], 5]).cuda()
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = torch.squeeze(out, 0)
        out = out.view(int(out.shape[0] / 8), 8, -1).unsqueeze(0)
        return out


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  #

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x


# ===========================================================


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        #         print("outch",out_channels)
        self.kernel_size = kernel_size
        # out channels为5*8=40
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        # print('A shape:', A.shape)
        # print('x before conv:', x.shape)
        x = self.conv(x)
        # print('x after conv:', x.shape)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # print('x after view:', x.shape)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        # print('x,A einsum:', x.shape)
        return x.contiguous(), A


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        #  print("outstg",out_channels)
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),

                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        # print('x original:', x.shape)
        res = self.residual(x)
        x, A = self.gcn(x, A)
        # print('x after gcn:', x.shape)
        # print('x before tcn:', x.shape)
        x = self.tcn(x) + res
        # print('x after tcn:', x.shape)        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class flow_conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super(flow_conv, self).__init__()
        flow_padding = (kernel_size - 1) // 2

        # print('flow_padding:',flow_padding)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              padding=flow_padding)

    def forward(self, flow):
        flow = self.conv(flow)
        return flow


class social_stgcnn(nn.Module):
    def __init__(self, n_stgcnn=3, n_txpcnn=1, input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3, flow_feat=50, n_fused_cnn=2, n_lstm_hidden=8,
                 use_image=True, use_flow=True, fusion_mode=1):
        super(social_stgcnn, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.n_fused_cnn = n_fused_cnn
        self.n_lstm_hidden = n_lstm_hidden
        self.use_image = use_image
        self.use_flow = use_flow
        self.fusion_mode = fusion_mode  # mode=0: mean ; mode=1: concatenation

        self.st_gcns_bbox = nn.ModuleList()
        self.st_gcns_bbox.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns_bbox.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        self.st_gcns_flow = nn.ModuleList()
        self.st_gcns_flow.append(st_gcn(flow_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns_flow.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        # image feature extraction
        self.vgg = VGG([1, 1, 1, 1, 1])
        self.lstm = LstmRNN(8, n_lstm_hidden, 8, 2)

        # fusion CNN
        self.fused_cnn = nn.ModuleList()
        if self.use_image and self.use_flow:
            self.fused_cnn.append(nn.Conv2d(15, 5, 3, padding=1))
        else:
            self.fused_cnn.append(nn.Conv2d(10, 5, 3, padding=1))

        for j in range(1, self.n_fused_cnn):
            self.fused_cnn.append(nn.Conv2d(5, 5, 3, padding=1))

        self.fusion_atv = nn.ModuleList()
        for j in range(self.n_fused_cnn):
            self.fusion_atv.append(nn.ReLU())

        # tpcnn
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.flow_conv = flow_conv(flow_feat, output_feat, kernel_size)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a, flow, img):
        # print('flow original:', flow.shape)

        if self.use_image:
            feat_img = self.vgg(img)  # out: [1,8,2,5]
            feat_img = feat_img.permute(2, 0, 1, 3)
            d = feat_img.shape[3]
            b = feat_img.shape[2]
            feat_img = feat_img.reshape(feat_img.shape[0], feat_img.shape[1], b * d)
            feat_img = feat_img.permute(2, 1, 0)
            feat_img = self.lstm(feat_img)
            feat_img = feat_img.permute(2, 1, 0)
            feat_img = feat_img.view(feat_img.shape[0], feat_img.shape[1], b, d)
            feat_img = feat_img.permute(1, 3, 0, 2)

        if self.use_flow:
            flow = flow.view(flow.shape[0], flow.shape[1], flow.shape[2], -1)
            flow = flow.view(flow.shape[0], flow.shape[3], flow.shape[2], flow.shape[1])
        for k in range(self.n_stgcnn):
            v, _ = self.st_gcns_bbox[k](v, a)
            if self.use_flow:
                flow, _ = self.st_gcns_flow[k](flow, a)

        # flow = self.flow_conv(flow)

        # v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        # flow = flow.view(flow.shape[0], flow.shape[2], flow.shape[1], flow.shape[3])

        # 融合方式1
        if self.fusion_mode == 0:
            # print(v.shape)
            # print(flow.shape)
            # print(feat_img.shape)
            fuse = (v + flow + feat_img) / 3

        # 融合方式2
        elif self.fusion_mode == 1:
            if self.use_flow and self.use_image:
                fuse = torch.cat([v, flow], dim=1)
                fuse = torch.cat([fuse, feat_img], dim=1)
                for k in range(self.n_fused_cnn):
                    fuse = self.fusion_atv[k](self.fused_cnn[k](fuse))

            elif self.use_flow and not self.use_image:
                fuse = torch.cat([v, flow], dim=1)
                for k in range(self.n_fused_cnn):
                    fuse = self.fusion_atv[k](self.fused_cnn[k](fuse))

            elif self.use_image and not self.use_flow:
                fuse = torch.cat([v, feat_img], dim=1)
                for k in range(self.n_fused_cnn):
                    fuse = self.fusion_atv[k](self.fused_cnn[k](fuse))

        fuse = fuse.permute(0, 2, 1, 3)
        # print('fused v:',v.shape)
        # txp
        # (1,8,5,2) -> (1,12,5,2)
        v = self.prelus[0](self.tpcnns[0](fuse))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        # print('Final Output:', v.shape)

        return v, a
