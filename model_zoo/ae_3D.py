import torch
from typing import Sequence
from copy import deepcopy as copy
from net_utils.activation_layers import CustomSwish
from net_utils.initialize import *
__all__ = ["Encoder", "Decoder", 'ConvAutoEncoder', 'DenseConvAutoEncoderBaur']


class Encoder(nn.Sequential):
    def __init__(self, in_channels: int, channels: Sequence[int], strides: Sequence[int],
                 kernel_size=5, norm='batch', act='leakyrelu', deconv_mode='upsample', name_prefix='shape'):
        """
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param name_prefix:
        :param bottleneck_size:
        """
        super(Encoder, self).__init__()
        padding = (kernel_size - 1) // 2
        layer_channels = in_channels
        encoder_channels = copy(channels)
        encoder_channels.append(channels[-1])
        for i, (c, s) in enumerate(zip(encoder_channels, strides)):
            stride = 1 if deconv_mode == 'stride_upsample' else s
            self.add_module(name_prefix + "_encode_%i" % i,
                            nn.Conv3d(in_channels=layer_channels, out_channels=c, kernel_size=kernel_size,
                                      stride=stride, padding=padding))#, dilation=(1,1), groups=1))
            if norm == 'batch':
                self.add_module(name_prefix + "_batch_%i" % i, nn.BatchNorm3d(c))
            if act == 'relu':
                self.add_module(name_prefix + "_act_%i" % i, nn.ReLU(True))
            elif act == 'leakyrelu':
                self.add_module(name_prefix + "_act_%i" % i, nn.LeakyReLU(True))
            else:
                self.add_module(name_prefix + "_act_%i" % i, CustomSwish())
            if deconv_mode == 'stride_upsample' and i!=len(channels)-1:
                self.add_module(name_prefix + "_max_pool%i" % i,
                                torch.nn.MaxPool3d(kernel_size=kernel_size, stride=s, padding=padding))
            layer_channels = c


class Decoder(nn.Sequential):
    def __init__(self, in_channels: int, channels: Sequence[int], out_ch: int, strides: Sequence[int],
                 kernel_size: int = 5, norm: str = 'batch', act: str = 'leakyrelu', deconv_mode='upsample',
                 decode_final: str = 'decode_final', bottleneck: bool = False, skip: bool = False, add_final: bool = True,
                 name_prefix: str = '_'):
        """
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        :param add_final:
        :param name_prefix:
        """
        super(Decoder, self).__init__()
        padding = (kernel_size - 1) // 2
        decode_channel_list = list(reversed(channels))
        decode_channel_list.append(channels[0])
        lr, nr_layers = 0, strides.count(2)
        decode_strides = strides[::-1] or [1]
        layer_channels = in_channels
        for i, (c, s) in enumerate(zip(decode_channel_list, decode_strides)):
            if i == 0 and bottleneck:
                continue
            if skip and s == 1:
                lr += 1
                layer_channels = layer_channels + channels[-i-1]

            if  deconv_mode == 'stride_upsample' and i!=0:
                self.add_module(name_prefix + "_upsample_%i" % i, nn.Upsample(scale_factor=s, mode='nearest'))
                self.add_module(name_prefix + "_decode_%i" % i,
                                nn.Conv3d(in_channels=layer_channels, out_channels=c, kernel_size=kernel_size
                                      , padding=padding))
            elif  deconv_mode == 'stride_upsample' and i==0:

                self.add_module(name_prefix + "_decode_%i" % i,
                                nn.Conv3d(in_channels=layer_channels, out_channels=c, kernel_size=kernel_size
                                      , padding=padding))
            elif  deconv_mode == 'stride':
                self.add_module(name_prefix + "_upsample_%i" % i, nn.Upsample(scale_factor=s, mode='nearest'))
                self.add_module(name_prefix + "_decode_%i" % i,
                                nn.Conv3d(in_channels=layer_channels, out_channels=c, kernel_size=kernel_size
                                      , padding=padding))
            else:
                self.add_module(name_prefix + "_decode_%i" % i,
                                nn.ConvTranspose3d(in_channels=layer_channels, out_channels=c,
                                                   kernel_size=kernel_size, stride=s, padding=padding,
                                                   output_padding=1))
            if norm == 'batch':
                self.add_module(name_prefix + "_batch_%i" % i, nn.BatchNorm3d(c))

            if act == 'relu':
                self.add_module(name_prefix + "_act_%i" % i, nn.ReLU(True))
            elif act == 'leakyrelu':
                self.add_module(name_prefix + "_act_%i" % i, nn.LeakyReLU(True))
            else:
                self.add_module(name_prefix + "_act_%i" % i, CustomSwish())

            layer_channels = c

        if add_final:
            self.add_module(name_prefix + "_decode_final",
                            nn.Conv3d(in_channels=layer_channels, out_channels=out_ch, kernel_size=1,
                                      ))
           # self.add_module(name_prefix + "_act_final", nn.Sigmoid())


class ConvAutoEncoder(nn.Module):

    def __init__(self, in_channels: int, channels: Sequence[int], out_ch: int, strides: Sequence[int],
                 kernel_size=3, norm='batch', act='leakyrelu', deconv_mode='upsample', act_final='sigmoid',
                 bottleneck=False, skip=False):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        """
        super(ConvAutoEncoder, self).__init__()

        self.encoder = Encoder(in_channels=in_channels, channels=channels, strides=strides,
                              kernel_size=kernel_size, norm=norm, act=act, deconv_mode=deconv_mode, name_prefix='conv_')

        self.decoder = Decoder(in_channels=channels[-1], channels=channels, out_ch=out_ch,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, deconv_mode=deconv_mode,
                              act_final=act_final, bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='conv_')

    # def forward(self,  x: torch):
    #     z = self.encoder(x)
    #     x_ = self.decoder(z)
    #     return x_, {'z': z}

    def forward(self, x:torch):
        encode_history, decode_history = [], []
        for i_e, enc_layer in enumerate(self.encoder):
            if i_e == 0:
                enc_x = enc_layer(x)
            else:
                enc_x = enc_layer(enc_x)
            if isinstance(enc_layer, CustomSwish):
            # if isinstance(enc_layer, nn.Conv2d):
                # if enc_x.shape[-1] != 1:
                encode_history.insert(0, enc_x)
        z = enc_x
        for i_d, dec_layer in enumerate(self.decoder):
            if i_d == 0:
                dec_x = dec_layer(enc_x)
            else:
                dec_x = dec_layer(dec_x)
            if isinstance(dec_layer, CustomSwish):
            # if isinstance(dec_layer, nn.Conv2d) or isinstance(dec_layer, nn.ConvTranspose2d):
                # if len(decode_history) < len(encode_history) and \
                #         encode_history[len(decode_history)].shape[-1] == dec_x.shape[-1]:
                decode_history.append(dec_x)
        return dec_x, {'z': z, 'encode_embeddings': encode_history, 'decode_embeddings': decode_history}


class DenseConvAutoEncoderBaur(nn.Module):
    """
    dense model implemented after https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI

    C. Baur, S. Denner, B. Wiestler, N. Navab, and S. Albarqouni. Autoencoders for unsupervised anomaly segmentation
    in brain mr images: a comparative study. Medical Image Analysis, 69: 101952, 2021.
    """

    def __init__(self, in_channels: int, channels: Sequence[int], out_ch: int, strides: Sequence[int],
                 kernel_size=5, norm='batch', act='leakyrelu', deconv_mode='upsample', act_final='sigmoid', bottleneck=False, skip=False):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        """
        super(DenseConvAutoEncoderBaur, self).__init__()

        self.encoder = Encoder(in_channels=in_channels, channels=channels, strides=strides,
                              kernel_size=kernel_size, norm=norm, act=act, deconv_mode=deconv_mode,name_prefix='conv_')

        self.decoder = Decoder(in_channels=channels[-1], channels=channels, out_ch=out_ch,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, deconv_mode=deconv_mode,
                              act_final=act_final, bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='conv_')
        self.lin_enc = nn.Conv3d(in_channels=channels[-1], out_channels=16, kernel_size=(1, 1), padding=0)
        self.lin_lay_enc = nn.Linear(1024, 128)
        self.lin_lay_dec = nn.Linear(128, 1024)
        self.lin_dec = nn.Conv3d(in_channels=16, out_channels=channels[-1], kernel_size=(1, 1), padding=0)

    def forward(self,  x: torch):
        z = self.encoder(x)
        z = self.lin_enc(z)
        z_shape = z.shape
        z = z.view(z_shape[0], -1)
        z = self.lin_lay_enc(z)
        z_ = self.lin_lay_dec(z)
        z_ = self.lin_dec(z_.view(z_shape))
        x_ = self.decoder(z_)
        return x_, {'z': z}

