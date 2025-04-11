import http.server
import json
import os
import queue
import random
import socketserver
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from requests.exceptions import RequestException

from VidIQ.autoencoder.dataset import (CNN_FRAME_NUM, IMAGE_EXT, VIDEO_DIR,
                                       ReshapeVideo, load_video_to_tensor)
from VidIQ.autoencoder.network2d import get_networks
from VidIQ.autoencoder.network3d import get_networks3d
from VidIQ.autoencoder.util import (DOWNLOAD_DIR, YUV_ENABLED, check_work_dir,
                                    save_compressed_data)
from VidIQ.dnn_model.inference import (prepare_imgs_inputs_for_video,
                                       run_dnn_task)
from VidIQ.super_resolution.models import get_models

check_work_dir()
with open('config.json', 'r') as f:
    config = json.load(f)

http_port = config['http_port']
server_ip = config['host']
server_port = config['port']
samplesize = config['samplesize']
head = config['filehead']
buff_size = config['buffsize']
dnn_task = config['task']
file_info_size = struct.calcsize(head)
rev = ReshapeVideo()

udp_port = 38792
sample_queue = queue.Queue()
result_num = 4
result_queue = queue.LifoQueue()
client_bandwidth = queue.Queue(maxsize=1)
old_server_decoder = queue.Queue(maxsize=1)
new_server_decoder = queue.Queue(maxsize=1)
server_com_index = queue.Queue()
server_sr_index = queue.Queue()


com_factor_num = 4
sr_multiplier_num = 4
default_com_index = 2
default_sr_index = 1


def tcp_recv(conn, DATA_DIR):
    info_size = 0
    file_info = b''
    while info_size < file_info_size:
        temp = conn.recv(file_info_size-info_size)
        file_info += temp
        info_size += len(temp)
    assert info_size == file_info_size
    file_name, file_size, send_t = struct.unpack(head, file_info)
    base_path = os.path.join(DATA_DIR, file_name.decode().replace('\x00', ''))
    data_path = base_path+'.pkl'
    received_size = 0
    chunk = b''
    while received_size < file_size:
        temp = conn.recv(file_size-received_size)
        chunk += temp
        received_size += len(temp)
    assert received_size == file_size
    with open(data_path, "wb") as f:
        f.write(chunk)
    transfer_time = time.time()-send_t
    assert transfer_time > 0
    bandwidth = 8*(file_info_size+file_size)/(1024**2) / transfer_time
    return base_path, data_path, bandwidth


def get_network_depth(com_index):
    return com_index+2


def get_fit_video_paths(fit_video_num=20):
    video_paths = [f.path for f in os.scandir(VIDEO_DIR) if f.is_dir()]
    assert fit_video_num <= len(video_paths)
    video_paths = random.sample(video_paths, fit_video_num)
    return video_paths


def get_scale_ratio(sr_index):
    return sr_index+1


def pipeline_benchmarks(video_paths, com_index, down_factor=1, sr_index=None):
    encoder3d, _, decoder3d, _ = get_networks3d(
        'eval', True, down_factor=down_factor, network_depth_3d=get_network_depth(com_index))

    if sr_index is not None:
        assert isinstance(sr_index, int)
        scale_ratio = get_scale_ratio(sr_index)
        if scale_ratio > 1:
            model3d, _ = get_models(
                'eval', True, network3d=True, scale_ratio=scale_ratio)

    acc_over_content = []
    for video_path in video_paths:
        video_tensor = load_video_to_tensor(
            video_path, length=CNN_FRAME_NUM, is_yuv=YUV_ENABLED).unsqueeze(0)
        with torch.no_grad():
            compressed_data = encoder3d(video_tensor)
            data_path = save_compressed_data(video_path, compressed_data)
            decoded_tensor = decoder3d(compressed_data)
            if sr_index is not None:
                if scale_ratio > 1:
                    decoded_tensor = model3d(decoded_tensor)
        _, inputs, _, gt_inputs = prepare_imgs_inputs_for_video(
            data_path, decoded_tensor, rev)
        acc_over_content.append(run_dnn_task(
            dnn_task, None, inputs, None, gt_inputs))

    return np.mean(acc_over_content)


def load_encoder_pool(network3d=True, cuda_enabled=False):
    encoder_pool = []
    for i in range(com_factor_num):
        if network3d:
            encoder, _ = get_networks3d(
                'eval', True, is_decoder=False, network_depth_3d=get_network_depth(i), cuda_enabled=cuda_enabled)
        else:
            encoder, _ = get_networks(
                'eval', True, is_decoder=False, network_depth=get_network_depth(i), cuda_enabled=cuda_enabled)
        encoder_pool.append(encoder)
    return encoder_pool


def load_decoder_pool(network3d=True, cuda_enabled=False):
    decoder_pool = []
    for i in range(com_factor_num):
        if network3d:
            decoder, _ = get_networks3d(
                'eval', True, is_encoder=False, network_depth_3d=get_network_depth(i), cuda_enabled=cuda_enabled)
        else:
            decoder, _ = get_networks(
                'eval', True, is_encoder=False, network_depth=get_network_depth(i), cuda_enabled=cuda_enabled)
        decoder_pool.append(decoder)
    return decoder_pool


def load_sr_model_pool(network3d=True, cuda_enabled=False):
    sr_model_pool = [None]
    for i in range(sr_multiplier_num):
        if i > 0:
            model, _ = get_models('eval', True, network3d=network3d,
                                  scale_ratio=get_scale_ratio(i), cuda_enabled=cuda_enabled)
            sr_model_pool.append(model)
    return sr_model_pool


class httpHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DOWNLOAD_DIR, **kwargs)


def start_http_server(port=http_port):
    server = socketserver.TCPServer(("", port), httpHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"HTTP Server started on port {port}")
    return server


def fetch_image(url):
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return transforms.ToTensor()(image)


def fetch_images_from_http_server(server_ip, image_paths, rank=0):
    start_time = time.time()
    try:
        with ThreadPoolExecutor() as executor:
            image_urls = [
                f"http://{server_ip}:{http_port}/{path}{IMAGE_EXT}" for path in image_paths]
            images = list(executor.map(fetch_image, image_urls))
    except RequestException as e:
        print('http server is closed')
        return None
    images = [img for img in images if img is not None]
    tensor = torch.stack(images, dim=0).to(rank)
    total_time = time.time() - start_time
    return tensor, total_time


if __name__ == "__main__":
    image_paths = ["val2017/000000000139", "val2017/000000344621",
                   "val2017/000000000285", "val2017/000000000632"]
    res = fetch_images_from_http_server(server_ip, image_paths)
    if res:
        tensor, total_time = res
        print(tensor.shape)
        print(total_time)
