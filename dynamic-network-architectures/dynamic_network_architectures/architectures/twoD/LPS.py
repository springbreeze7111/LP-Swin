import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from collections.abc import Sequence
from dynamic_network_architectures.architectures.encoder.swin import SwinTransformer
import numpy as np
import torch
from dynamic_network_architectures.architectures.modules.lib.modules import *
from dynamic_network_architectures.building_blocks.block import *
from dynamic_network_architectures.architectures.modules.lib.res2net import *


class ResBlock(nn.Module):
    def __init__(
        self,

        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int] ,
        norm_name: tuple ,
        act_name: tuple  = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple  = None,
        spatial_dims: int=2,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class upsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4),
            Conv(in_channels // 4, out_channels)
            # ResBlock(in_channels, in_channels // 4, 3, 1, norm_name="instance"),
            # ResBlock(in_channels // 4, out_channels, 3, 1, norm_name="instance"),

        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                          stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x
        return x


class LPS(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LPS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.swinViT = SwinTransformer(
                                      patch_size=(2,2),
                                      in_chans=n_channels,
                                      embed_dim=64,
                                      depths=[2, 2, 6, 2],
                                      num_heads=[4, 8, 16, 32],
                                      window_size=(7, 7),
                                      mlp_ratio=4.,
                                      qkv_bias=True,
                                     spatial_dims=2
        )
        # resnet = res2net50_v1b_26w_4s(pretrained=False)
        # self.encoder1_conv = resnet.conv1
        # self.encoder1_bn = resnet.bn1
        # self.encoder1_relu = resnet.relu
        # self.maxpool = resnet.maxpool
        # self.encoder2 = resnet.layer1
        # self.encoder3 = resnet.layer2
        # self.encoder4 = resnet.layer3
        # self.encoder5 = resnet.layer4


        # self.x5_dem_1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        # self.x4_dem_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        # self.x3_dem_1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        # self.x2_dem_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_dem_1 = nn.Sequential(ResBlock(1024, 512, kernel_size=3,
                                      stride=1,
                                      norm_name="instance"),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),

                                      )
        self.x4_dem_1 = nn.Sequential(ResBlock(512, 256, kernel_size=3,
                                      stride=1,
                                      norm_name="instance"),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(ResBlock(256, 128, kernel_size=3,
                                      stride=1,
                                      norm_name="instance"),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(ResBlock(128, 64, kernel_size=3,
                                      stride=1,
                                      norm_name="instance"),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.upsample5 = nn.Sequential(
            Conv(512, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.upsample4 = upsample(768, 256)
        self.upsample3 = upsample(384, 128)
        self.upsample2 = upsample(192, 64)
        self.upsample1 = upsample(128, 64)

        self.GFB1 = GFB(64)
        self.GFB2 = GFB(64)
        self.GFB3 = GFB(128)
        self.GFB4 = GFB(256)

        self.out5 = Out(512, 256)
        self.out4 = Out(256, 128)
        self.out3 = Out(128, 64)
        self.out2 = Out(64, 64)
        self.out1 = Out(64, self.n_classes)
        # self.out5 = nn.Sequential(ResBlock(512, 256, kernel_size=3,
        #                               stride=1,
        #                               norm_name="instance"),
        #                               nn.BatchNorm2d(256),
        #                               nn.ReLU(inplace=True),
        #                               )
        # self.out4 = nn.Sequential(ResBlock(256, 128, kernel_size=3,
        #                               stride=1,
        #                               norm_name="instance"),
        #                               nn.BatchNorm2d(128),
        #                               nn.ReLU(inplace=True),
        #                               )
        # self.out3 = nn.Sequential(ResBlock(128, 64, kernel_size=3,
        #                               stride=1,
        #                               norm_name="instance"),
        #                               nn.BatchNorm2d(64),
        #                               nn.ReLU(inplace=True),
        #                               )
        # self.out2 = nn.Sequential(ResBlock(64, 64, kernel_size=3,
        #                               stride=1,
        #                               norm_name="instance"),
        #                               nn.BatchNorm2d(64),
        #                               nn.ReLU(inplace=True),
        #                               )
        # self.out1 = nn.Sequential(ResBlock(64, n_classes, kernel_size=3,
        #                               stride=1,
        #                               norm_name="instance"),
        #                               nn.BatchNorm2d(n_classes),
        #                               nn.ReLU(inplace=True),
        #                               )


    def forward(self, x):
        edge_feature = make_laplace_pyramid(x, 4, 4)
        edge_feature = edge_feature[1]

        # e1 = self.encoder1_conv(x)
        # e1 = self.encoder1_bn(e1)
        # e1 = self.encoder1_relu(e1)
        # e1_pool = self.maxpool(e1)
        #
        # e2 = self.encoder2(e1_pool)
        #
        # e3 = self.encoder3(e2)
        #
        # e4 = self.encoder4(e3)
        #
        # e5 = self.encoder5(e4)
        #Swin encoder
        e1, e2, e3, e4, e5 = self.swinViT(x)
        e5_dem_1 = self.x5_dem_1(e5)
        e4_dem_1 = self.x4_dem_1(e4)
        e3_dem_1 = self.x3_dem_1(e3)
        e2_dem_1 = self.x2_dem_1(e2)

        # Decoder
        d5 = self.upsample5(e5_dem_1)
        out5 = self.out5(d5)
        GFB4 = self.GFB4(edge_feature, e4_dem_1, out5)

        d4 = self.upsample4(d5, GFB4)
        out4 = self.out4(d4)
        GFB3 = self.GFB3(edge_feature, e3_dem_1, out4)

        d3 = self.upsample3(d4, GFB3)
        out3 = self.out3(d3)
        GFB2 = self.GFB2(edge_feature, e2_dem_1, out3)

        d2 = self.upsample2(d3, GFB2)
        out2 = self.out2(d2)
        GFB1 = self.GFB1(edge_feature, e1, out2)

        d1 = self.upsample1(d2, GFB1)
        out1 = self.out1(d1)
        return out1



def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] = 3,
    stride: Sequence[int]  = 1,
    act: tuple = Act.PRELU,
    norm: tuple  = Norm.INSTANCE,
    dropout: tuple = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(kernel_size: Sequence[int], stride: Sequence[int]) -> tuple[int, ...]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size, stride, padding
) -> tuple[int, ...] :
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


if __name__ == '__main__':
    from fvcore.nn import parameter_count_table
    # 将模型移动到主 GPU
    model = LPS(4, 3)
    # # 生成输入数据并移动到主 GPU
    # x = torch.randn(1, 4, 128, 128)
    # # 前向传播
    # x = model(x)
    #
    # # 打印模型参数和输出形状
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters in ex model: {total_params}")
    # print(parameter_count_table(model))
    # print(x.shape)
    #
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Param: {name}, size: {param.size()}, stride: {param.stride()}")

    for i, (name, layer) in enumerate(model.named_modules()):
        if name != "":  # 跳过最外层（整个模型本身）
            print(f"{i}: {name}")