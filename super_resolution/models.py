import os
import re
import shutil
import subprocess
import time
from multiprocessing.pool import ThreadPool

import torch
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from VidIQ.autoencoder.dataset import (IMAGE_DIR, IMAGE_HEIGHT, IMAGE_WIDTH,
                                        ReshapeVideo)
from VidIQ.autoencoder.util import (CUDA_ENABLED, PROJECT_DIR, WEIGHTS_DIR,
                                     ColorConverter)

# MODEL_NAME = "srcnn"
MODEL_NAME = "edsr"
SCALE_RATIO = 2

LR_IMAGE_DIR = IMAGE_DIR+'_lrx'+str(SCALE_RATIO)
model_dir = WEIGHTS_DIR+'/'+MODEL_NAME


def scale_image(img_name, ratio=1/SCALE_RATIO):
    image = Image.open(IMAGE_DIR+'/'+img_name)
    resized_img = image.resize(
        (int(IMAGE_WIDTH*ratio), int(IMAGE_HEIGHT*ratio)), Image.BICUBIC)
    resized_img.save(LR_IMAGE_DIR+'/'+img_name)


def downscale_image_dataset():
    subprocess.call(['bash', PROJECT_DIR+'/video_data/clean.sh'])
    if not os.path.exists(LR_IMAGE_DIR):
        os.mkdir(LR_IMAGE_DIR)
        start = time.time()
        image_filenames = [file for file in os.listdir(IMAGE_DIR)]
        frame_number = len(image_filenames)
        # Use thread pools to process all images in parallel
        pool = ThreadPool()
        pool.map(scale_image, image_filenames)
        pool.close()
        pool.join()
        print(f'Downscaling of {frame_number} images completed')
        end = time.time()
        print(f'Running time {(end-start)/60}m')


def get_depth_and_channels(network3d=False):
    '''
    Get network depth and bits channels of Encoder
    '''
    if network3d:
        network_dir = PROJECT_DIR+'/autoencoder/network3d.py'
        name1 = '^NETWORK_DEPTH_3D'
        name2 = '^BITS_CHANNEL_3D'
    else:
        network_dir = PROJECT_DIR+'/autoencoder/network2d.py'
        name1 = '^NETWORK_DEPTH'
        name2 = '^BITS_CHANNEL'
    network_depth = int(subprocess.check_output(
        f"grep -m 1 {name1} {network_dir} | grep -oP '{name1}\s*=\s*\K[^,]*'", shell=True).decode().strip())
    bits_channels = int(subprocess.check_output(
        f"grep -m 1 {name2} {network_dir} | grep -oP '{name2}\s*=\s*\K[^,]*'", shell=True).decode().strip())
    return network_depth, bits_channels


def upscale2d(integrated, network_depth, channels, scale_ratio):
    upscale = nn.Sequential()
    if integrated:
        factor = 2**(network_depth)
        upscale.add_module("convt", nn.ConvTranspose2d(
            channels, 3, factor, factor))
    upscale.add_module("color", ColorConverter())
    upscale.add_module("conv", nn.Conv2d(
        3, 3 * (scale_ratio**2), kernel_size=3, padding=1))
    upscale.add_module("pixel", nn.PixelShuffle(scale_ratio))
    return upscale


def upscale3d(integrated, network_depth, channels, scale_ratio):
    upscale = nn.Sequential()
    if integrated:
        factor = 2**(network_depth)
        upscale.add_module("convt", nn.ConvTranspose3d(channels, 3, kernel_size=(
            network_depth+1, factor, factor), stride=(1, factor, factor)))
    rev = ReshapeVideo()
    upscale.add_module("reshape1", rev)
    upscale.add_module("color", ColorConverter())
    upscale.add_module("conv", nn.Conv2d(
        3, 3 * (scale_ratio**2), kernel_size=3, padding=1))
    upscale.add_module("pixel", nn.PixelShuffle(scale_ratio))
    upscale.add_module("reshape2", rev)
    return upscale


class SRCNN(nn.Module):
    def __init__(self, integrated, network3d, scale_ratio):
        super(SRCNN, self).__init__()
        self.depth, self.channels = get_depth_and_channels(network3d)
        if network3d:
            self.upscale = upscale3d(
                integrated, self.depth, self.channels, scale_ratio)
            self.conv1 = nn.Conv3d(3, 64, kernel_size=9, padding=2)
            self.conv2 = nn.Conv3d(64, 32, kernel_size=1, padding=2)
            self.conv3 = nn.Conv3d(32, 3, kernel_size=5, padding=2)
        else:
            self.upscale = upscale2d(
                integrated, self.depth, self.channels, scale_ratio)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=2)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
            self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.upscale(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = nn.Sigmoid()(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, network3d):
        super(ResidualBlock, self).__init__()
        if network3d:
            self.conv1 = nn.Conv3d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class EDSR(nn.Module):
    def __init__(self, num_blocks, num_channels, integrated, network3d, scale_ratio):
        super(EDSR, self).__init__()
        self.depth, self.channels = get_depth_and_channels(network3d)
        if network3d:
            self.upscale = upscale3d(
                integrated, self.depth, self.channels, scale_ratio)
            self.head = nn.Conv3d(3, num_channels, kernel_size=3, padding=1)
            self.mid = nn.Conv3d(num_channels, num_channels,
                                 kernel_size=3, padding=1)
            self.tail = nn.Conv3d(num_channels, 3, kernel_size=3, padding=1)
        else:
            self.upscale = upscale2d(
                integrated, self.depth, self.channels, scale_ratio)
            self.head = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
            self.mid = nn.Conv2d(num_channels, num_channels,
                                 kernel_size=3, padding=1)
            self.tail = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels, network3d) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.upscale(x)
        out = self.head(x)
        residual = out
        out = self.residual_blocks(out)
        out = self.mid(out)
        out += residual
        out = self.tail(out)
        out = nn.Sigmoid()(out)
        return out


def get_weights_path(model_dir, integrated, network3d, transfer, scale_ratio):
    if transfer:
        transfer_model_dir = model_dir + '_x'+transfer
    model_dir += ('_x'+str(scale_ratio))
    if integrated:
        if network3d:
            loss_index = subprocess.check_output(
                f"grep -m 1 '^loss_index' {PROJECT_DIR}'/joint_training/train3d.py' | grep -oP '^loss_index\s*=\s*\K[^,]*'", shell=True).decode().strip()
            depth, channels = get_depth_and_channels(True)
            model_dir += ('.' + str(depth)+'.'+str(channels) + '.3d')
        else:
            loss_index = subprocess.check_output(
                f"grep -m 1 '^loss_index' {PROJECT_DIR}'/joint_training/train.py' | grep -oP '^loss_index\s*=\s*\K[^,]*'", shell=True).decode().strip()
            depth, channels = get_depth_and_channels(False)
            model_dir += ('.' + str(depth)+'.'+str(channels))
    else:
        if network3d:
            loss_index = subprocess.check_output(
                f"grep -m 1 '^loss_index' {PROJECT_DIR}'/super_resolution/train3d.py' | grep -oP '^loss_index\s*=\s*\K[^,]*'", shell=True).decode().strip()
        else:
            loss_index = subprocess.check_output(
                f"grep -m 1 '^loss_index' {PROJECT_DIR}'/super_resolution/train.py' | grep -oP '^loss_index\s*=\s*\K[^,]*'", shell=True).decode().strip()
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if transfer:
        model_path = transfer_model_dir
    else:
        model_path = model_dir
    if integrated:
        model_path += ('/integrated_loss_'+loss_index+'.pth')
    else:
        if network3d:
            model_path += ('/loss_'+loss_index+'_3d.pth')
        else:
            model_path += ('/loss_'+loss_index+'.pth')
    return model_path


def get_weights_from_path(rank, model, path, tran_num, transfer):
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
        scale_ratio = int(results[0])
        if tran_num == 1:
            target = str(SCALE_RATIO)
            path = re.sub('_x'+str(scale_ratio), '_x'+target, path)
        elif tran_num == 3:
            NETWORK_DEPTH, BITS_CHANNELS = get_depth_and_channels(False)
            target = str(SCALE_RATIO)+'.'+str(NETWORK_DEPTH) + \
                '.'+str(BITS_CHANNELS)
            network_depth = int(results[1])
            bits_channel = int(results[2])
            path = re.sub('_x'+str(scale_ratio)+'.'+str(network_depth) +
                          '.'+str(bits_channel), '_x'+target, path)
        elif tran_num == 4:
            NETWORK_DEPTH, BITS_CHANNELS = get_depth_and_channels(True)
            target = str(SCALE_RATIO)+'.'+str(NETWORK_DEPTH) + \
                '.'+str(BITS_CHANNELS)+'.3d'
            network_depth = int(results[1])
            bits_channel = int(results[2])
            path = re.sub('_x'+str(scale_ratio)+'.'+str(network_depth) +
                          '.'+str(bits_channel)+'.3d', '_x'+target, path)
        remove_layers = []
        if tran_num == 1:
            if scale_ratio == SCALE_RATIO:
                return state_dict, path
        else:
            if (scale_ratio == SCALE_RATIO) and (network_depth == NETWORK_DEPTH)\
               and (bits_channel == BITS_CHANNELS):
                return state_dict, path
            if network_depth != NETWORK_DEPTH or bits_channel != BITS_CHANNELS:
                remove_layers.append('upscale.convt.weight')
                remove_layers.append('upscale.convt.bias')
        if scale_ratio != SCALE_RATIO:
            remove_layers.append('upscale.conv.weight')
            remove_layers.append('upscale.conv.bias')
            remove_layers.append('upscale.pixel.weight')
            remove_layers.append('upscale.pixel.bias')
        transfer_state_dict = model.state_dict()
        for key, _ in state_dict.items():
            if key not in remove_layers:
                transfer_state_dict[key] = state_dict[key]
        if rank == 0:
            print(f'Using transfer learning (from x{transfer} to x{target})')
        return transfer_state_dict, path


def get_models(mode='train', is_load=False, rank=0, integrated=False, network3d=False, transfer=None,
               scale_ratio=SCALE_RATIO, cuda_enabled=CUDA_ENABLED):
    '''
    mode -> Set model mode to 'train' or 'eval'
    is_load -> Whether to load srcnn/edsr weights 
               for continued training or evaluation
    transfer -> Load weights of other network structures
                for transfer learning (e.g., "2" or "2.4.8" or "2.4.16.3d" )
    '''
    tran_num = 0
    if transfer:
        assert is_load, 'When transfer is not None is_load must be True'
        results = transfer.split('.')
        tran_num = len(results)
        assert tran_num == 1 or tran_num == 3 or tran_num == 4
        if tran_num == 1:
            assert (not integrated)
        elif tran_num == 3:
            assert integrated and (not network3d)
        elif tran_num == 4:
            assert integrated and network3d

    if MODEL_NAME == "srcnn":
        model = SRCNN(integrated, network3d, scale_ratio)
    elif MODEL_NAME == "edsr":
        model = EDSR(2, 64, integrated, network3d, scale_ratio)

    model_path = get_weights_path(
        model_dir, integrated, network3d, transfer, scale_ratio)
    if is_load:
        state_dict, model_path = get_weights_from_path(
            rank, model, model_path, tran_num, transfer)
        model.load_state_dict(state_dict)
    else:
        if rank == 0:
            print(f'Save weights to {model_path}')

    if mode == 'train':
        model = model.to(rank)
        model = DDP(model)
        model.train()
    elif mode == 'eval':
        if cuda_enabled:
            model = model.to(rank)
        model.eval()

    return model, model_path


if __name__ == '__main__':
    downscale_image_dataset()
