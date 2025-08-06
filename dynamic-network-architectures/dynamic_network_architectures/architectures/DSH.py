from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.block import *
from dynamic_network_architectures.building_blocks.residual import *

class DSH(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.OMC = ChannelAggregationFFN(embed_dims=2)
        self.soft_wavelette = Soft_WTConv3d(in_channels=2, out_channels=2, kernel_size=3, stride=1, bias=True, wt_levels=1, wt_type='db1')
        self.encoder = Encoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                                               strides,
                                               n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                               dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                               nonlin_first=nonlin_first)
        self.bottleneck = LRC(conv_op, 768)
        self.decoder = Dencoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                          nonlin_first=nonlin_first)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        print('using custom')

    def forward(self, x):
        x1 = x[:, [0, 2], :, :, :]
        x2 = x[:, [1, 3], :, :, :]
        # x1 = self.OMC(x1)
        x1 = self.soft_wavelette(x1)
        x = torch.cat([x1, x2], dim=1)
        skips = self.encoder(x)
        skips[-1] = self.bottleneck(skips[-1])
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class Encoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            # print(x.shape)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output

class Dencoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks2(
                n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            # print(x.shape)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
            # r = seg_outputs[0]
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output



class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):

        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, input_channels, output_channels[0], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs-1)
            ],
            ShuffleAtt(
                conv_op, output_channels[-2], output_channels[-1], kernel_size, 1, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
            ),
            *[
                ShuffleAtt(
                    conv_op, output_channels[-2], output_channels[-1], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                )
                for i in range(1, num_convs-1)
            ],
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(
            self.initial_stride), "just give the image size without color/feature channels or " \
                                  "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                  "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output



class StackedConvBlocks2(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):

        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                # , nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, input_channels, output_channels[0], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                    # , nonlin_first
                )
                for i in range(1, num_convs-1)
            ],
            ConvDropoutNormReLU(
                conv_op, output_channels[-2], output_channels[-1], kernel_size, 1, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                # ,nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[-2], output_channels[-1], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                    # , nonlin_first
                )
                for i in range(1, num_convs-1)
            ],

        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(
            self.initial_stride), "just give the image size without color/feature channels or " \
                                  "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                  "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class ConvDropoutNormReLU2(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ConvDropoutNormReLU2, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)

class LRC(nn.Module):
    def __init__(self, conv_op=torch.nn.modules.conv.Conv3d, channels: int=768):
        super().__init__()
        self.is_3d = True
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.is_3d = False
            self.ch_wv = nn.Conv2d(channels, channels // 2, kernel_size=(1, 1))
            self.ch_wq = nn.Conv2d(channels, 1, kernel_size=(1, 1))
            self.ch_wz = nn.Conv2d(channels // 2, channels, kernel_size=(1, 1))
            self.sp_wv = nn.Conv2d(channels, channels // 2, kernel_size=(1, 1))
            self.sp_wq = nn.Conv2d(channels, channels // 2, kernel_size=(1, 1))
            self.agp = nn.AdaptiveAvgPool2d((1, 1))
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.ch_wv = nn.Conv3d(channels, channels // 2, kernel_size=(1, 1, 1))
            self.ch_wq = nn.Conv3d(channels, 1, kernel_size=(1, 1, 1))
            self.ch_wz = nn.Conv3d(channels // 2, channels, kernel_size=(1, 1, 1))
            self.sp_wv = nn.Conv3d(channels, channels // 2, kernel_size=(1, 1, 1))
            self.sp_wq = nn.Conv3d(channels, channels // 2, kernel_size=(1, 1, 1))
            self.agp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ln = nn.LayerNorm(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w, d = x.size()
        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1).unsqueeze(-1)  # bs,c//2,1,1, 1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0,2,1).reshape(b, c, 1, 1, 1)  # bs,c,1,1, 1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1, 1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w*d
        spatial_wq = spatial_wq.permute(0, 2, 3, 4, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w*d
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w, d))  # bs,1,h,w
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out

class OMC(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, 92, kernel_size, stride, conv_bias,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.SEAttention = SEAttention(conv_op, channel=92)
        self.conv2 = ConvDropoutNormReLU(conv_op, 92, output_channels, kernel_size, 1, conv_bias, norm_op,
                                         norm_op_kwargs, None, None, None, None)

        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x


        # Stochastic Depth
        self.apply_stochastic_depth = False if stochastic_depth_p == 0.0 else True
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)

        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(self.output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8)

        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1, False, norm_op,
                                        norm_op_kwargs, None, None, None, None
                                        )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.SEAttention(out)
        out = self.conv2(out)
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out += residual
        return self.nonlin2(out)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        output_size_conv1 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # conv2
        output_size_conv2 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # skip conv (if applicable)
        if (self.input_channels != self.output_channels) or any([i != j for i, j in zip(input_size, size_after_stride)]):
            assert isinstance(self.skip, nn.Sequential)
            output_size_skip = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        else:
            assert not isinstance(self.skip, nn.Sequential)
            output_size_skip = 0
        return output_size_conv1 + output_size_conv2 + output_size_skip


def create_wavelet_filter_3d(wave='db1', in_channels=1, out_channels=1, dtype=torch.float):
    """
    创建适用于3D小波分解的卷积核（卷积形式实现离散小波变换 DWT）。
    输出: (wt_filter, iwt_filter)，维度为 (C, 1, 2, 2, 2)，共8个子带卷积核。
    """
    # 获取1D小波滤波器
    dec_lo, dec_hi = pywt.Wavelet(wave).dec_lo, pywt.Wavelet(wave).dec_hi
    rec_lo, rec_hi = pywt.Wavelet(wave).rec_lo, pywt.Wavelet(wave).rec_hi

    # 转为张量
    dec_lo = torch.tensor(dec_lo, dtype=dtype)
    dec_hi = torch.tensor(dec_hi, dtype=dtype)
    rec_lo = torch.tensor(rec_lo, dtype=dtype)
    rec_hi = torch.tensor(rec_hi, dtype=dtype)

    # 生成3D滤波器：8个子带（LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH）
    def outer3(a, b, c):
        return torch.einsum("i,j,k->ijk", a, b, c)

    # 构建分解核（DWT）与重建核（IWT）
    filters = []
    inv_filters = []

    for fz in [dec_lo, dec_hi]:
        for fy in [dec_lo, dec_hi]:
            for fx in [dec_lo, dec_hi]:
                kernel = outer3(fz, fy, fx)
                filters.append(kernel)

    for fz in [rec_lo, rec_hi]:
        for fy in [rec_lo, rec_hi]:
            for fx in [rec_lo, rec_hi]:
                kernel = outer3(fz, fy, fx)
                inv_filters.append(kernel)

    # [8, D, H, W]
    filters = torch.stack(filters, dim=0)
    inv_filters = torch.stack(inv_filters, dim=0)

    # 适配in_channels和out_channels：depthwise形式 (C, 1, D, H, W)
    wt_filter = filters.unsqueeze(1).repeat(in_channels, 1, 1, 1, 1)
    iwt_filter = inv_filters.unsqueeze(1).repeat(in_channels, 1, 1, 1, 1)

    return wt_filter, iwt_filter



class Soft_WTConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(Soft_WTConv3d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        # 3D Wavelet filters（需自定义）
        self.wt_filter, self.iwt_filter = create_wavelet_filter_3d(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform_3d, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform_3d, filters=self.iwt_filter)

        # 主干卷积
        self.base_conv = nn.Conv3d(in_channels, in_channels, kernel_size, padding='same', stride=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1, 1])

        # 高频路径：深度可分离卷积
        self.wavelet_convs = nn.ModuleList([
            nn.Conv3d(in_channels * 8, in_channels * 8, kernel_size, padding='same', stride=1, groups=in_channels * 8, bias=False)
            for _ in range(self.wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 8, 1, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

        self.soft_thr_lambda = nn.Parameter(torch.tensor(0.05), requires_grad=True)

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv3d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def soft_threshold(self, x, lambd):
        return torch.sign(x) * torch.relu(torch.abs(x) - lambd)

    def forward(self, x):
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            pad_d = curr_shape[2] % 2
            pad_h = curr_shape[3] % 2
            pad_w = curr_shape[4] % 2
            if pad_d or pad_h or pad_w:
                curr_pads = (0, pad_w, 0, pad_h, 0, pad_d)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)  # [B, C, 8, D/2, H/2, W/2]
            curr_x_ll = curr_x[:, :, 0, :, :, :]  # LL

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 8, shape_x[3], shape_x[4], shape_x[5])

            curr_x_tag = self.wavelet_convs[i](curr_x_tag)

            # Soft thresholding
            curr_x_tag = curr_x_tag.reshape(shape_x)  # [B, C, 8, D, H, W]
            ll = curr_x_tag[:, :, 0:1, :, :, :]
            hf = curr_x_tag[:, :, 1:, :, :, :]
            hf = self.soft_threshold(hf, self.soft_thr_lambda)
            curr_x_tag = torch.cat([ll, hf], dim=2)
            curr_x_tag = curr_x_tag.reshape(shape_x[0], shape_x[1] * 8, shape_x[3], shape_x[4], shape_x[5])
            curr_x_tag = self.wavelet_scale[i](curr_x_tag)
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:, :, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3], :curr_shape[4]]

        x_tag = next_x_ll

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride:
            x = self.do_stride(x)

        return x

def wavelet_transform_3d(x, filters):
    """
    3D小波分解
    输入:
        x: 输入张量 [B, C, D, H, W]
        filters: 小波卷积核，维度为 [C, 1, 2, 2, 2] * 8 共 8*C 个
    输出:
        out: 小波分解结果 [B, C, 8, D//2, H//2, W//2]
    """
    B, C, D, H, W = x.shape
    device = x.device

    # 展开所有filter: [C*8, 1, 2, 2, 2]
    wt_filters = filters.view(-1, 1, 2, 2, 2).to(device)  # 共 C*8 个卷积核

    # 输入 reshape 为 [1, B*C, D, H, W]，分组卷积每组8个filter
    x = x.view(1, B * C, D, H, W)
    out = F.conv3d(x, wt_filters, stride=2, padding=0, groups=B * C)  # [1, B*C*8, D//2, H//2, W//2]
    out = out.view(B, C, 8, out.shape[-3], out.shape[-2], out.shape[-1])
    return out

def inverse_wavelet_transform_3d(x, filters):
    """
    3D小波重建（反变换）
    输入:
        x: 小波系数 [B, C, 8, D, H, W]
        filters: 重建卷积核 [C, 1, 2, 2, 2] * 8
    输出:
        out: 重建后的张量 [B, C, D*2, H*2, W*2]
    """
    B, C, n_subbands, D, H, W = x.shape
    device = x.device

    x = x.view(B * C, 8, D, H, W)
    x = x.view(B * C * 8, 1, D, H, W)

    iwt_filters = filters.view(-1, 1, 2, 2, 2).to(device)  # [C*8, 1, 2, 2, 2]

    out = F.conv_transpose3d(x, iwt_filters, stride=2, padding=0, groups=B * C)
    out = out.view(B, C, out.shape[-3], out.shape[-2], out.shape[-1])
    return out


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


if __name__ == '__main__':
    import torch
    from fvcore.nn import parameter_count_table, FlopCountAnalysis
    x = nn.Parameter(torch.randn(1, 4, 128, 128, 128))
    model = DSH(4, 5,[48, 96, 192, 384, 768], conv_op=nn.Conv3d,
                kernel_sizes= [
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ]
                    ],
                strides=[
                        [
                            1,
                            1,
                            1
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ]
                    ],
                n_conv_per_stage=[
                        2,
                        2,
                        2,
                        2,
                        2
                    ],
                n_conv_per_stage_decoder=[
                        2,
                        2,
                        2,
                        2
                    ], num_classes=3)
    x = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in ex model: {total_params/1000000}")
    # print(parameter_count_table(model))
    # print(x.shape)