import math
import os
import re
import shutil
import subprocess
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from VidIQ.autoencoder.dataset import (CNN_FRAME_NUM, IMAGE_HEIGHT,
                                        IMAGE_WIDTH, MIN_FRAME_NUM,
                                        ReshapeVideo)
from VidIQ.autoencoder.util import (CUDA_ENABLED, PROJECT_DIR, WEIGHTS_DIR,
                                     YUV_ENABLED, convert_image_colorspace)

NETWORK_DEPTH_3D = 4
assert NETWORK_DEPTH_3D >= 2 and NETWORK_DEPTH_3D <= 5
FEATURE_CHANNEL_3D = 2
assert FEATURE_CHANNEL_3D == 2 or FEATURE_CHANNEL_3D == 4 or FEATURE_CHANNEL_3D == 8
BITS_CHANNEL_3D = 16
assert BITS_CHANNEL_3D >= 2 and BITS_CHANNEL_3D <= 32


def print_autoencoder3d_info():
    print('Network parameters of 3d autoencoder')
    print(
        f'NETWORK_DEPTH_3D={NETWORK_DEPTH_3D} (Strong feature representation but longer training time)')
    print(
        f'FEATURE_CHANNEL_3D={FEATURE_CHANNEL_3D} (Beneficial to feature extraction but consumes more GPU memory)')
    print(
        f'BITS_CHANNEL_3D={BITS_CHANNEL_3D} (Retain more features (higher accuracy target) but compressed data is larger)')
    print(f'CNN_FRAME_NUM={CNN_FRAME_NUM}')


def get_layer_channels3d(network_depth_3d, feature_channel_3d, bits_channel_3d):
    layer2_channels = bits_channel_3d if network_depth_3d == 2 else 64*feature_channel_3d
    layer3_channels = bits_channel_3d if network_depth_3d == 3 else 128*feature_channel_3d
    layer4_channels = bits_channel_3d if network_depth_3d == 4 else 256*feature_channel_3d
    return layer2_channels, layer3_channels, layer4_channels


class VideoEncoder(nn.Module):
    def __init__(self, down_factor=1, network_depth_3d=NETWORK_DEPTH_3D,
                 feature_channel_3d=FEATURE_CHANNEL_3D, bits_channel_3d=BITS_CHANNEL_3D):
        super(VideoEncoder, self).__init__()
        self.network_depth_3d = network_depth_3d
        layer2_channels, layer3_channels, layer4_channels, = get_layer_channels3d(
            network_depth_3d, feature_channel_3d, bits_channel_3d)
        self.conv_par = [
            [3, 32*feature_channel_3d,
                (3, 3, 3), (1, 1, 1), (1, 1, 1)],
            [32*feature_channel_3d, layer2_channels,
                (3, 3, 3), (1, 1, 1), (1, 1, 1)],
            [64*feature_channel_3d, layer3_channels,
                (3, 3, 3), (1, 1, 1), (1, 1, 1)],
            [128*feature_channel_3d, layer4_channels,
                (3, 3, 3), (1, 1, 1), (1, 1, 1)],
            [256*feature_channel_3d, bits_channel_3d,
             (3, 3, 3), (1, 1, 1), (1, 1, 1)]]
        self.max_par = [
            [(2, 2, 2), (1, 2, 2)],
            [(2, 2, 2), (1, 2, 2)],
            [(2, 2, 2), (1, 2, 2)],
            [(2, 2, 2), (1, 2, 2)],
            [(2, 2, 2), (1, 2, 2)]]
        self.down = nn.Upsample(scale_factor=(1, 1/down_factor, 1/down_factor))
        self.conv1 = nn.Conv3d(self.conv_par[0][0], self.conv_par[0][1], kernel_size=self.conv_par[0]
                               [2], stride=self.conv_par[0][3], padding=self.conv_par[0][4])
        self.re1 = nn.ReLU(True)
        self.max1 = nn.MaxPool3d(
            kernel_size=self.max_par[0][0], stride=self.max_par[0][1])
        self.conv2 = nn.Conv3d(self.conv_par[1][0], self.conv_par[1][1], kernel_size=self.conv_par[1]
                               [2], stride=self.conv_par[1][3], padding=self.conv_par[1][4])
        self.bm2 = nn.BatchNorm3d(self.conv_par[1][1])
        self.re2 = nn.ReLU(True)
        self.max2 = nn.MaxPool3d(
            kernel_size=self.max_par[1][0], stride=self.max_par[1][1])
        if network_depth_3d >= 3:
            self.conv3 = nn.Conv3d(self.conv_par[2][0], self.conv_par[2][1], kernel_size=self.conv_par[2]
                                   [2], stride=self.conv_par[2][3], padding=self.conv_par[2][4])
            self.bm3 = nn.BatchNorm3d(self.conv_par[2][1])
            self.re3 = nn.ReLU(True)
            self.max3 = nn.MaxPool3d(
                kernel_size=self.max_par[2][0], stride=self.max_par[2][1])
        if network_depth_3d >= 4:
            self.conv4 = nn.Conv3d(self.conv_par[3][0], self.conv_par[3][1], kernel_size=self.conv_par[3]
                                   [2], stride=self.conv_par[3][3], padding=self.conv_par[3][4])
            self.bm4 = nn.BatchNorm3d(self.conv_par[3][1])
            self.re4 = nn.ReLU(True)
            self.max4 = nn.MaxPool3d(
                kernel_size=self.max_par[3][0], stride=self.max_par[3][1])
        if network_depth_3d == 5:
            self.conv5 = nn.Conv3d(self.conv_par[4][0], self.conv_par[4][1], kernel_size=self.conv_par[4]
                                   [2], stride=self.conv_par[4][3], padding=self.conv_par[4][4])
            self.bm5 = nn.BatchNorm3d(self.conv_par[4][1])
            self.re5 = nn.ReLU(True)
            self.max5 = nn.MaxPool3d(
                kernel_size=self.max_par[4][0], stride=self.max_par[4][1])

    def forward(self, x):
        assert x.shape[2] >= MIN_FRAME_NUM
        x = self.down(x)
        x = self.conv1(x)
        x = self.re1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.bm2(x)
        x = self.re2(x)
        x = self.max2(x)
        if self.network_depth_3d >= 3:
            x = self.conv3(x)
            x = self.bm3(x)
            x = self.re3(x)
            x = self.max3(x)
        if self.network_depth_3d >= 4:
            x = self.conv4(x)
            x = self.bm4(x)
            x = self.re4(x)
            x = self.max4(x)
        if self.network_depth_3d == 5:
            x = self.conv5(x)
            x = self.bm5(x)
            x = self.re5(x)
            x = self.max5(x)
        return x


class VideoDecoder(nn.Module):
    def __init__(self, joint=False, network_depth_3d=NETWORK_DEPTH_3D,
                 feature_channel_3d=FEATURE_CHANNEL_3D, bits_channel_3d=BITS_CHANNEL_3D):
        super(VideoDecoder, self).__init__()
        self.network_depth_3d = network_depth_3d
        layer2_channels, layer3_channels, layer4_channels, = get_layer_channels3d(
            network_depth_3d, feature_channel_3d, bits_channel_3d)
        self.convt_par = [
            [bits_channel_3d, 256*feature_channel_3d,
                (4, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)],
            [layer4_channels, 128*feature_channel_3d,
                (4, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)],
            [layer3_channels, 64*feature_channel_3d,
                (4, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)],
            [layer2_channels, 32*feature_channel_3d,
                (4, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)],
            [32*feature_channel_3d, 3,
             (4, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)]]
        if network_depth_3d == 5:
            self.convt5 = nn.ConvTranspose3d(self.convt_par[0][0], self.convt_par[0][1], kernel_size=self.convt_par[0]
                                             [2], stride=self.convt_par[0][3], padding=self.convt_par[0][4], output_padding=self.convt_par[0][5])
            self.re5 = nn.ReLU(True)
        if network_depth_3d >= 4:
            self.convt4 = nn.ConvTranspose3d(self.convt_par[1][0], self.convt_par[1][1], kernel_size=self.convt_par[1]
                                             [2], stride=self.convt_par[1][3], padding=self.convt_par[1][4], output_padding=self.convt_par[1][5])
            self.re4 = nn.ReLU(True)
        if network_depth_3d >= 3:
            self.convt3 = nn.ConvTranspose3d(self.convt_par[2][0], self.convt_par[2][1], kernel_size=self.convt_par[2]
                                             [2], stride=self.convt_par[2][3], padding=self.convt_par[2][4], output_padding=self.convt_par[2][5])
            self.re3 = nn.ReLU(True)
        self.convt2 = nn.ConvTranspose3d(self.convt_par[3][0], self.convt_par[3][1], kernel_size=self.convt_par[3]
                                         [2], stride=self.convt_par[3][3], padding=self.convt_par[3][4], output_padding=self.convt_par[3][5])
        self.re2 = nn.ReLU(True)
        self.convt1 = nn.ConvTranspose3d(self.convt_par[4][0], self.convt_par[4][1], kernel_size=self.convt_par[4]
                                         [2], stride=self.convt_par[4][3], padding=self.convt_par[4][4], output_padding=self.convt_par[4][5])
        self.sigm = nn.Sigmoid()
        self.joint = joint
        self.rev = ReshapeVideo()

    def forward(self, x):
        if self.network_depth_3d == 5:
            x = self.convt5(x)
            x = self.re5(x)
        if self.network_depth_3d >= 4:
            x = self.convt4(x)
            x = self.re4(x)
        if self.network_depth_3d >= 3:
            x = self.convt3(x)
            x = self.re3(x)
        x = self.convt2(x)
        x = self.re2(x)
        x = self.convt1(x)
        x = self.sigm(x)
        if not self.joint:
            if YUV_ENABLED:
                x = self.rev(x)
                x = convert_image_colorspace(x, False)
                x = self.rev(x)
        return x


def get_weights3d_path(weights3d_dir, joint, transfer):
    if transfer:
        weights3d_dir = WEIGHTS_DIR+'/weights_network3d_'+transfer
    if joint:
        loss_index = subprocess.check_output(
            f"grep -m 1 '^loss_index' {PROJECT_DIR}'/joint_training/train3d.py' | grep -oP '^loss_index\s*=\s*\K[^,]*'", shell=True).decode().strip()
        encoder3d_path = weights3d_dir+'/joint_encoder_loss_'+loss_index+'.pth'
        decoder3d_path = weights3d_dir+'/joint_decoder_loss_'+loss_index+'.pth'
    else:
        loss_index = subprocess.check_output(
            f"grep -m 1 '^loss_index' {PROJECT_DIR}'/autoencoder/train3d.py' | grep -oP '^loss_index\s*=\s*\K[^,]*'", shell=True).decode().strip()
        encoder3d_path = weights3d_dir+'/encoder_loss_'+loss_index+'.pth'
        decoder3d_path = weights3d_dir+'/decoder_loss_'+loss_index+'.pth'
    return encoder3d_path, decoder3d_path


def get_weights3d_from_path(rank, network, path, transfer):
    if rank == 0:
        if not os.path.exists(path):
            src_path = re.sub(r'loss_\d+', 'loss_2', path)
            shutil.copy(src_path, path)
            print(f'Copy weights from {src_path}')
        print(f'Load weights from {path}')
    time.sleep(rank)
    state_dict3d = torch.load(path)
    if not transfer:
        return state_dict3d, path
    else:
        results = transfer.split('.')
        network_depth = int(results[0])
        feature_channel = int(results[1])
        bits_channel = int(results[2])
        cnn_frame_num = int(results[3])
        assert feature_channel == FEATURE_CHANNEL_3D
        assert cnn_frame_num == CNN_FRAME_NUM
        path = re.sub(str(network_depth) + '.'+str(feature_channel) + '.'+str(bits_channel)+'.'+str(cnn_frame_num),
                      str(NETWORK_DEPTH_3D) + '.'+str(FEATURE_CHANNEL_3D) + '.'+str(BITS_CHANNEL_3D)+'.'+str(CNN_FRAME_NUM), path)
        if (network_depth == NETWORK_DEPTH_3D) and (bits_channel == BITS_CHANNEL_3D):
            return state_dict3d, path
        network_depth = min(network_depth, NETWORK_DEPTH_3D)-1
        assert network_depth >= 1
        transfer_state_dict3d = network.state_dict()
        for key, _ in state_dict3d.items():
            number = int(re.search(r'\d+', key).group())
            if number <= network_depth:
                transfer_state_dict3d[key] = state_dict3d[key]
        if rank == 0:
            print(
                f'Using transfer learning (from {transfer} to {str(NETWORK_DEPTH_3D)}.{str(FEATURE_CHANNEL_3D)}.{str(BITS_CHANNEL_3D)}.{str(CNN_FRAME_NUM)})')
        return transfer_state_dict3d, path


def get_networks3d(mode='train', is_load=False, rank=0, down_factor=1, is_encoder=True, is_decoder=True, joint=False, transfer=None,
                   network_depth_3d=NETWORK_DEPTH_3D, feature_channel_3d=FEATURE_CHANNEL_3D, bits_channel_3d=BITS_CHANNEL_3D, cuda_enabled=CUDA_ENABLED):
    '''
    mode -> Set network mode to 'train' or 'eval'
    is_load -> Whether to load 3d encoder/decoder weights 
               for continued training or evaluation
    transfer -> Load weights of other network structures
                for transfer learning (e.g., "4.2.16.6")
    '''
    assert is_encoder or is_decoder, 'is_encoder and is_decoder cannot both be False'
    if transfer:
        assert is_load, 'When transfer is not None is_load must be True'

    if is_encoder:
        encoder = VideoEncoder(down_factor, network_depth_3d,
                               feature_channel_3d, bits_channel_3d)
    if is_decoder:
        decoder = VideoDecoder(joint, network_depth_3d,
                               feature_channel_3d, bits_channel_3d)

    # Weight path for encoder and decoder
    weights3d_dir = WEIGHTS_DIR+'/weights_network3d_' + str(network_depth_3d) + '.'+str(
        feature_channel_3d) + '.'+str(bits_channel_3d)+'.'+str(CNN_FRAME_NUM)
    if not os.path.exists(weights3d_dir):
        os.mkdir(weights3d_dir)
    encoder3d_path, decoder3d_path = get_weights3d_path(
        weights3d_dir, joint, transfer)

    if is_load:
        if is_encoder:
            state_dict3d, encoder3d_path = get_weights3d_from_path(
                rank, encoder, encoder3d_path, transfer)
            encoder.load_state_dict(state_dict3d)
        if is_decoder:
            state_dict3d, decoder3d_path = get_weights3d_from_path(
                rank, decoder, decoder3d_path, transfer)
            decoder.load_state_dict(state_dict3d)
    else:
        if rank == 0:
            if is_encoder:
                print(f'Save weights to {encoder3d_path}')
            if is_decoder:
                print(f'Save weights to {decoder3d_path}')

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
        return encoder, encoder3d_path, decoder, decoder3d_path
    if is_encoder and (not is_decoder):
        return encoder, encoder3d_path
    if (not is_encoder) and is_decoder:
        return decoder, decoder3d_path


def conv3d_output_shape(input_shape, output_channels, kernel_size, stride, padding):
    batch_size, input_channels, input_depth, input_height, input_width = input_shape
    output_depth = math.floor(
        (input_depth - kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
    output_height = math.floor(
        (input_height - kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
    output_width = math.floor(
        (input_width - kernel_size[2] + 2 * padding[2]) / stride[2]) + 1
    return (batch_size, output_channels, output_depth, output_height, output_width)


def maxpool3d_output_shape(input_shape, kernel_size, stride):
    batch_size, input_channels, input_depth, input_height, input_width = input_shape
    output_depth = math.floor((input_depth - kernel_size[0]) / stride[0]) + 1
    output_height = math.floor((input_height - kernel_size[1]) / stride[1]) + 1
    output_width = math.floor((input_width - kernel_size[2]) / stride[2]) + 1
    return (batch_size, input_channels, output_depth, output_height, output_width)


def conv_transpose3d_output_shape(input_shape, output_channels, kernel_size, stride, padding, output_padding):
    batch_size, input_channels, input_depth, input_height, input_width = input_shape
    output_depth = math.floor(
        (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0])
    output_height = math.floor(
        (input_height - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1])
    output_width = math.floor(
        (input_width - 1) * stride[2] - 2 * padding[2] + kernel_size[2] + output_padding[2])
    return (batch_size, output_channels, output_depth, output_height, output_width)


def show_shape_transform3d(total_layers, batch_size):
    encoder = VideoEncoder()
    conv_par = encoder.conv_par
    max_par = encoder.max_par
    convt_par = VideoDecoder().convt_par
    tensor_shape = (batch_size, 3, CNN_FRAME_NUM, IMAGE_HEIGHT, IMAGE_WIDTH)
    print(f'{tensor_shape}')
    for _ in range(NETWORK_DEPTH_3D):
        print(f'--------VideoEncoder.conv3d{_+1}--------')
        tensor_shape = conv3d_output_shape(
            tensor_shape, conv_par[_][1], conv_par[_][2], conv_par[_][3], conv_par[_][4])
        print(f'{tensor_shape}')
        print(f'--------VideoEncoder.max3d{_+1}--------')
        tensor_shape = maxpool3d_output_shape(
            tensor_shape, max_par[_][0], max_par[_][1])
        print(f'{tensor_shape}')
    index = total_layers-NETWORK_DEPTH_3D
    for _ in range(index, total_layers):
        print(f'--------VideoDecoder.convt3d{total_layers-_}--------')
        tensor_shape = conv_transpose3d_output_shape(
            tensor_shape, convt_par[_][1], convt_par[_][2], convt_par[_][3], convt_par[_][4], convt_par[_][5])
        print(f'{tensor_shape}')


if __name__ == '__main__':
    show_shape_transform3d(5, 2)
