import json
import os
import time

import torch

from VidIQ.autoencoder.dataset import (IMAGE_DIR, IMAGE_EXT, IMAGE_HEIGHT,
                                        IMAGE_WIDTH, VIDEO_DIR,
                                        load_video_to_tensor)
from VidIQ.autoencoder.network2d import get_networks, print_autoencoder_info
from VidIQ.autoencoder.network3d import (get_networks3d,
                                          print_autoencoder3d_info)
from VidIQ.autoencoder.util import (CUDA_ENABLED, IMAGE_EXT, PROJECT_DIR,
                                     YUV_ENABLED, convert_image_to_tensor,
                                     load_compressed_data, print_device_info,
                                     save_compressed_data)
from VidIQ.dnn_model.faster_rcnn import load_faster_rcnn
from VidIQ.super_resolution.models import (LR_IMAGE_DIR, MODEL_NAME,
                                            SCALE_RATIO, get_models)

print_device_info()
print_autoencoder_info()
print_autoencoder3d_info()
print(f'SR model is {MODEL_NAME}x{SCALE_RATIO}')

encoder, _, decoder, _ = get_networks('eval', True)
encoder_joint, _, decoder_joint, _ = get_networks('eval', True, joint=True)
encoder3d, _, decoder3d, _ = get_networks3d('eval', True)
encoder3d_joint, _, decoder3d_joint, _ = get_networks3d(
    'eval', True, joint=True)
model, _ = get_models('eval', True)
model_integrated, _ = get_models('eval', True, integrated=True)
model3d, _ = get_models('eval', True, network3d=True)
model3d_integrated, _ = get_models(
    'eval', True, network3d=True, integrated=True)
weights, dnn_task = load_faster_rcnn()

with open(PROJECT_DIR+'/research_work/journal_3/metric.json', 'r') as f:
    config = json.load(f)
client_scale = config['client_scale']
server_scale = config['server_scale']


def generate_image_list(start_kbyte, kbyte_step, list_length):
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(IMAGE_EXT)]
    sorted_images = sorted(
        image_files, key=lambda f: os.path.getsize(os.path.join(IMAGE_DIR, f)))
    result_list = []
    kbyte_list = []
    for i in range(list_length):
        target_kbyte = start_kbyte + i * kbyte_step
        def size_key(f): return abs(os.path.getsize(
            os.path.join(IMAGE_DIR, f)) / 1024 - target_kbyte)
        nearest_image = min(sorted_images, key=size_key)
        result_list.append(os.path.splitext(nearest_image)[0])
        kbyte_list.append(os.path.getsize(
            os.path.join(IMAGE_DIR, nearest_image)) / 1024)
    return result_list, kbyte_list


def VidIQ_image(image_path):
    img = image_path + IMAGE_EXT
    img_tensor = convert_image_to_tensor(
        img, YUV_ENABLED, CUDA_ENABLED).unsqueeze(0)

    with torch.no_grad():
        compressed_data = encoder(img_tensor)
    data_path = save_compressed_data(img, compressed_data)
    t = time.time()
    compressed_data = load_compressed_data(data_path)
    with torch.no_grad():
        sr_tensor = model_integrated(compressed_data)
    decoding_time = time.time()-t

    t = time.time()
    with torch.no_grad():
        compressed_data = encoder_joint(img_tensor)
    data_path = save_compressed_data(img, compressed_data)
    encoding_time = time.time()-t
    kbyte = os.path.getsize(data_path)/1024
    compressed_data = load_compressed_data(data_path)
    with torch.no_grad():
        decoded_tensor = decoder_joint(compressed_data)
        sr_tensor = model(decoded_tensor)
    return decoded_tensor, sr_tensor, kbyte, encoding_time, decoding_time


def VidIQ_video(video_directory, length):
    video_tensor = load_video_to_tensor(
        video_directory, length=length, is_yuv=YUV_ENABLED,
        resize=[IMAGE_HEIGHT//SCALE_RATIO, IMAGE_WIDTH//SCALE_RATIO]).unsqueeze(0)

    with torch.no_grad():
        compressed_data = encoder3d(video_tensor)
    data_path = save_compressed_data(video_directory, compressed_data)
    t = time.time()
    compressed_data = load_compressed_data(data_path)
    with torch.no_grad():
        sr_video_tensor = model3d_integrated(compressed_data)
    decoding_time = time.time()-t

    t = time.time()
    with torch.no_grad():
        compressed_data = encoder3d_joint(video_tensor)
    data_path = save_compressed_data(video_directory, compressed_data)
    encoding_time = time.time()-t
    kbyte = os.path.getsize(data_path)/1024
    compressed_data = load_compressed_data(data_path)
    with torch.no_grad():
        decoded_tensor = decoder3d_joint(compressed_data)
        sr_video_tensor = model3d(decoded_tensor)
    return decoded_tensor, sr_video_tensor, kbyte, encoding_time, decoding_time


image_path = LR_IMAGE_DIR+'/000000013291'
decoded_tensor, sr_tensor, kbyte, encoding_time, decoding_time = VidIQ_image(
    image_path)
print(kbyte, encoding_time, decoding_time)
video_directory = VIDEO_DIR+'/dog_0014_021'
decoded_tensor, sr_video_tensor, kbyte, encoding_time, decoding_time = VidIQ_video(
    video_directory, 8)
print(kbyte, encoding_time, decoding_time)
result_list, kbyte_list = generate_image_list(600, 10, 10)
print(result_list, kbyte_list)
