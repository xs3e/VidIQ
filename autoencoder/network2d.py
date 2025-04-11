import math
import os
import re
import shutil
import subprocess
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from VidIQ.autoencoder.dataset import IMAGE_HEIGHT, IMAGE_WIDTH
from VidIQ.autoencoder.util import (CUDA_ENABLED, PROJECT_DIR, WEIGHTS_DIR,
                                     YUV_ENABLED, convert_image_colorspace)

NETWORK_DEPTH = 4
assert NETWORK_DEPTH >= 2 and NETWORK_DEPTH <= 5
FEATURE_CHANNEL = 8
assert FEATURE_CHANNEL == 2 or FEATURE_CHANNEL == 4 or FEATURE_CHANNEL == 8
BITS_CHANNEL = 8
assert BITS_CHANNEL >= 2 and BITS_CHANNEL <= 32


def print_autoencoder_info():
    print('Network parameters of autoencoder')
    print(
        f'NETWORK_DEPTH={NETWORK_DEPTH} (Strong feature representation but longer training time)')
    print(
        f'FEATURE_CHANNEL={FEATURE_CHANNEL} (Beneficial to feature extraction but consumes more GPU memory)')
    print(
        f'BITS_CHANNEL={BITS_CHANNEL} (Retain more features (higher accuracy target) but compressed data is larger)')


def get_layer_channels(network_depth, feature_channel, bits_channel):
    layer2_channels = bits_channel if network_depth == 2 else 64*feature_channel
    layer3_channels = bits_channel if network_depth == 3 else 128*feature_channel
    layer4_channels = bits_channel if network_depth == 4 else 256*feature_channel
    return layer2_channels, layer3_channels, layer4_channels


class Encoder(nn.Module):
    def __init__(self, down_factor=1, network_depth=NETWORK_DEPTH,
                 feature_channel=FEATURE_CHANNEL, bits_channel=BITS_CHANNEL):
        super(Encoder, self).__init__()
        self.network_depth = network_depth
        layer2_channels, layer3_channels, layer4_channels = get_layer_channels(
            network_depth, feature_channel, bits_channel)
        self.conv_par = [
            [3, 32*feature_channel, 3, 1, 1],
            [32*feature_channel, layer2_channels, 3, 1, 1],
            [64*feature_channel, layer3_channels, 3, 1, 1],
            [128*feature_channel, layer4_channels, 3, 1, 1],
            [256*feature_channel, bits_channel, 3, 1, 1]]
        self.max_par = [
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2]]
        self.down = nn.Upsample(scale_factor=1/down_factor)
        self.conv1 = nn.Conv2d(self.conv_par[0][0], self.conv_par[0][1], kernel_size=self.conv_par[0]
                               [2], stride=self.conv_par[0][3], padding=self.conv_par[0][4])
        self.re1 = nn.ReLU(True)
        self.max1 = nn.MaxPool2d(
            kernel_size=self.max_par[0][0], stride=self.max_par[0][1])
        self.conv2 = nn.Conv2d(self.conv_par[1][0], self.conv_par[1][1], kernel_size=self.conv_par[1]
                               [2], stride=self.conv_par[1][3], padding=self.conv_par[1][4])
        self.bm2 = nn.BatchNorm2d(self.conv_par[1][1])
        self.re2 = nn.ReLU(True)
        self.max2 = nn.MaxPool2d(
            kernel_size=self.max_par[1][0], stride=self.max_par[1][1])
        if network_depth >= 3:
            self.conv3 = nn.Conv2d(self.conv_par[2][0], self.conv_par[2][1], kernel_size=self.conv_par[2]
                                   [2], stride=self.conv_par[2][3], padding=self.conv_par[2][4])
            self.bm3 = nn.BatchNorm2d(self.conv_par[2][1])
            self.re3 = nn.ReLU(True)
            self.max3 = nn.MaxPool2d(
                kernel_size=self.max_par[2][0], stride=self.max_par[2][1])
        if network_depth >= 4:
            self.conv4 = nn.Conv2d(self.conv_par[3][0], self.conv_par[3][1], kernel_size=self.conv_par[3]
                                   [2], stride=self.conv_par[3][3], padding=self.conv_par[3][4])
            self.bm4 = nn.BatchNorm2d(self.conv_par[3][1])
            self.re4 = nn.ReLU(True)
            self.max4 = nn.MaxPool2d(
                kernel_size=self.max_par[3][0], stride=self.max_par[3][1])
        if network_depth == 5:
            self.conv5 = nn.Conv2d(self.conv_par[4][0], self.conv_par[4][1], kernel_size=self.conv_par[4]
                                   [2], stride=self.conv_par[4][3], padding=self.conv_par[4][4])
            self.bm5 = nn.BatchNorm2d(self.conv_par[4][1])
            self.re5 = nn.ReLU(True)
            self.max5 = nn.MaxPool2d(
                kernel_size=self.max_par[4][0], stride=self.max_par[4][1])

    def forward(self, x):
        x = self.down(x)
        x = self.conv1(x)
        x = self.re1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.bm2(x)
        x = self.re2(x)
        x = self.max2(x)
        if self.network_depth >= 3:
            x = self.conv3(x)
            x = self.bm3(x)
            x = self.re3(x)
            x = self.max3(x)
        if self.network_depth >= 4:
            x = self.conv4(x)
            x = self.bm4(x)
            x = self.re4(x)
            x = self.max4(x)
        if self.network_depth == 5:
            x = self.conv5(x)
            x = self.bm5(x)
            x = self.re5(x)
            x = self.max5(x)
        return x


class Decoder(nn.Module):
    def __init__(self,  joint=False, network_depth=NETWORK_DEPTH,
                 feature_channel=FEATURE_CHANNEL, bits_channel=BITS_CHANNEL,):
        super(Decoder, self).__init__()
        self.network_depth = network_depth
        layer2_channels, layer3_channels, layer4_channels = get_layer_channels(
            network_depth, feature_channel, bits_channel)
        self.convt_par = [
            [bits_channel, 256*feature_channel, 3, 2, 1, 1],
            [layer4_channels, 128*feature_channel, 3, 2, 1, 1],
            [layer3_channels, 64*feature_channel, 3, 2, 1, 1],
            [layer2_channels, 32*feature_channel, 3, 2, 1, 1],
            [32*feature_channel, 3, 3, 2, 1, 1]]
        if network_depth == 5:
            self.convt5 = nn.ConvTranspose2d(self.convt_par[0][0], self.convt_par[0][1], kernel_size=self.convt_par[0]
                                             [2], stride=self.convt_par[0][3], padding=self.convt_par[0][4], output_padding=self.convt_par[0][5])
            self.re5 = nn.ReLU(True)
        if network_depth >= 4:
            self.convt4 = nn.ConvTranspose2d(self.convt_par[1][0], self.convt_par[1][1], kernel_size=self.convt_par[1]
                                             [2], stride=self.convt_par[1][3], padding=self.convt_par[1][4], output_padding=self.convt_par[1][5])
            self.re4 = nn.ReLU(True)
        if network_depth >= 3:
            self.convt3 = nn.ConvTranspose2d(self.convt_par[2][0], self.convt_par[2][1], kernel_size=self.convt_par[2]
                                             [2], stride=self.convt_par[2][3], padding=self.convt_par[2][4], output_padding=self.convt_par[2][5])
            self.re3 = nn.ReLU(True)
        self.convt2 = nn.ConvTranspose2d(self.convt_par[3][0], self.convt_par[3][1], kernel_size=self.convt_par[3]
                                         [2], stride=self.convt_par[3][3], padding=self.convt_par[3][4], output_padding=self.convt_par[3][5])
        self.re2 = nn.ReLU(True)
        self.convt1 = nn.ConvTranspose2d(self.convt_par[4][0], self.convt_par[4][1], kernel_size=self.convt_par[4]
                                         [2], stride=self.convt_par[4][3], padding=self.convt_par[4][4], output_padding=self.convt_par[4][5])
        self.sigm = nn.Sigmoid()
        self.joint = joint

    def forward(self, x):
        if self.network_depth == 5:
            x = self.convt5(x)
            x = self.re5(x)
        if self.network_depth >= 4:
            x = self.convt4(x)
            x = self.re4(x)
        if self.network_depth >= 3:
            x = self.convt3(x)
            x = self.re3(x)
        x = self.convt2(x)
        x = self.re2(x)
        x = self.convt1(x)
        x = self.sigm(x)
        if not self.joint:
            if YUV_ENABLED:
                x = convert_image_colorspace(x, False)
        return x


def get_weights_path(weights2d_dir, joint, transfer):
    if transfer:
        weights2d_dir = WEIGHTS_DIR+'/weights_network_'+transfer
    if joint:
        loss_index = subprocess.check_output(
            f"grep -m 1 '^loss_index' {PROJECT_DIR}'/joint_training/train.py' | grep -oP '^loss_index\s*=\s*\K[^,]*'", shell=True).decode().strip()
        encoder_path = weights2d_dir+'/joint_encoder_loss_'+loss_index+'.pth'
        decoder_path = weights2d_dir+'/joint_decoder_loss_'+loss_index+'.pth'
    else:
        loss_index = subprocess.check_output(
            f"grep -m 1 '^loss_index' {PROJECT_DIR}'/autoencoder/train.py' | grep -oP '^loss_index\s*=\s*\K[^,]*'", shell=True).decode().strip()
        encoder_path = weights2d_dir+'/encoder_loss_'+loss_index+'.pth'
        decoder_path = weights2d_dir+'/decoder_loss_'+loss_index+'.pth'
    return encoder_path, decoder_path


def get_weights_from_path(rank, network, path, transfer):
    if rank == 0:
        if not os.path.exists(path):
            src_path = re.sub(r'loss_\d+', 'loss_2', path)
            shutil.copy(src_path, path)
            print(f'Copy weights from {src_path}')
        print(f'Load weights from {path}')
    time.sleep(rank)
    state_dict = torch.load(path)
    if not transfer:
        return state_dict, path
    else:
        results = transfer.split('.')
        network_depth = int(results[0])
        feature_channel = int(results[1])
        bits_channel = int(results[2])
        assert feature_channel == FEATURE_CHANNEL
        path = re.sub(str(network_depth)+'.'+str(feature_channel)+'.'+str(bits_channel),
                      str(NETWORK_DEPTH)+'.'+str(FEATURE_CHANNEL)+'.'+str(BITS_CHANNEL), path)
        if (network_depth == NETWORK_DEPTH) and (bits_channel == BITS_CHANNEL):
            return state_dict, path
        network_depth = min(network_depth, NETWORK_DEPTH)-1
        assert network_depth >= 1
        transfer_state_dict = network.state_dict()
        for key, _ in state_dict.items():
            number = int(re.search(r'\d+', key).group())
            if number <= network_depth:
                transfer_state_dict[key] = state_dict[key]
        if rank == 0:
            print(
                f'Using transfer learning (from {transfer} to {str(NETWORK_DEPTH)}.{str(FEATURE_CHANNEL)}.{str(BITS_CHANNEL)})')
        return transfer_state_dict, path


def get_networks(mode='train', is_load=False, rank=0, down_factor=1, is_encoder=True, is_decoder=True,  joint=False, transfer=None,
                 network_depth=NETWORK_DEPTH, feature_channel=FEATURE_CHANNEL, bits_channel=BITS_CHANNEL, cuda_enabled=CUDA_ENABLED):
    '''
    mode -> Set network mode to 'train' or 'eval'
    is_load -> Whether to load encoder/decoder weights 
               for continued training or evaluation
    transfer -> Load weights of other network structures
                for transfer learning (e.g., "4.8.8")
    '''
    assert is_encoder or is_decoder, 'is_encoder and is_decoder cannot both be False'
    if transfer:
        assert is_load, 'When transfer is not None is_load must be True'

    if is_encoder:
        encoder = Encoder(down_factor, network_depth,
                          feature_channel, bits_channel)
    if is_decoder:
        decoder = Decoder(joint, network_depth, feature_channel, bits_channel)

    # Weight path for encoder and decoder
    weights2d_dir = WEIGHTS_DIR+'/weights_network_' + \
        str(network_depth)+'.'+str(feature_channel) + '.'+str(bits_channel)
    if not os.path.exists(weights2d_dir):
        os.mkdir(weights2d_dir)
    encoder_path, decoder_path = get_weights_path(
        weights2d_dir, joint, transfer)

    if is_load:
        if is_encoder:
            state_dict, encoder_path = get_weights_from_path(
                rank, encoder, encoder_path, transfer)
            encoder.load_state_dict(state_dict)
        if is_decoder:
            state_dict, decoder_path = get_weights_from_path(
                rank, decoder, decoder_path, transfer)
            decoder.load_state_dict(state_dict)
    else:
        if rank == 0:
            if is_encoder:
                print(f'Save weights to {encoder_path}')
            if is_decoder:
                print(f'Save weights to {decoder_path}')

    if mode == 'train':
        if is_encoder:
            encoder = encoder.to(rank)
            encoder = DDP(encoder)
            encoder.train()
        if is_decoder:
            decoder = decoder.to(rank)
            decoder = DDP(decoder)
            decoder.train()
    elif mode == 'eval':
        if cuda_enabled:
            if is_encoder:
                encoder = encoder.to(rank)
            if is_decoder:
                decoder = decoder.to(rank)
        if is_encoder:
            encoder.eval()
        if is_decoder:
            decoder.eval()

    if is_encoder and is_decoder:
        return encoder, encoder_path, decoder, decoder_path
    if is_encoder and (not is_decoder):
        return encoder, encoder_path
    if (not is_encoder) and is_decoder:
        return decoder, decoder_path


def conv_output_shape(input_shape, output_channels, kernel_size, stride, padding):
    batch_size, input_channels, input_height, input_width = input_shape
    output_height = math.floor(
        (input_height - kernel_size + 2*padding) / stride) + 1
    output_width = math.floor(
        (input_width - kernel_size + 2*padding) / stride) + 1
    return (batch_size, output_channels, output_height, output_width)


def maxpool_output_shape(input_shape, kernel_size, stride):
    batch_size, input_channels, input_height, input_width = input_shape
    output_height = math.floor((input_height - kernel_size) / stride) + 1
    output_width = math.floor((input_width - kernel_size) / stride) + 1
    return (batch_size, input_channels, output_height, output_width)


def conv_transpose_output_shape(input_shape, output_channels, kernel_size, stride, padding, output_padding):
    batch_size, input_channels, input_height, input_width = input_shape
    output_height = math.floor(
        (input_height - 1) * stride - 2 * padding + kernel_size + output_padding)
    output_width = math.floor(
        (input_width - 1) * stride - 2 * padding + kernel_size + output_padding)
    return (batch_size, output_channels, output_height, output_width)


def show_shape_transform(total_layers, batch_size):
    encoder = Encoder()
    conv_par = encoder.conv_par
    max_par = encoder.max_par
    convt_par = Decoder().convt_par
    tensor_shape = (batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    print(f'{tensor_shape}')
    for _ in range(NETWORK_DEPTH):
        print(f'--------Encoder.conv{_+1}--------')
        tensor_shape = conv_output_shape(
            tensor_shape, conv_par[_][1], conv_par[_][2], conv_par[_][3], conv_par[_][4])
        print(f'{tensor_shape}')
        print(f'--------Encoder.max{_+1}--------')
        tensor_shape = maxpool_output_shape(
            tensor_shape, max_par[_][0], max_par[_][1])
        print(f'{tensor_shape}')
    index = total_layers-NETWORK_DEPTH
    for _ in range(index, total_layers):
        print(f'--------Decoder.convt{total_layers-_}--------')
        tensor_shape = conv_transpose_output_shape(
            tensor_shape, convt_par[_][1], convt_par[_][2], convt_par[_][3], convt_par[_][4], convt_par[_][5])
        print(f'{tensor_shape}')


if __name__ == '__main__':
    show_shape_transform(5, 8)
