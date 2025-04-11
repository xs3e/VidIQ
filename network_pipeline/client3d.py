import os
import socket
import struct
import time

import torch

from VidIQ.autoencoder.dataset import (CNN_FRAME_NUM, VIDEO_DIR,
                                       load_video_to_tensor)
from VidIQ.autoencoder.util import save_compressed_data
from VidIQ.network_pipeline.util import (head, load_encoder_pool, samplesize,
                                         server_ip, server_port)

encoder3d_pool = load_encoder_pool()
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

i = 0
for video_name in os.listdir(VIDEO_DIR):
    if i >= samplesize:
        break
    video_path = os.path.join(VIDEO_DIR, video_name)
    if os.path.isdir(video_path):
        video_tensor = load_video_to_tensor(
            video_path, length=CNN_FRAME_NUM).unsqueeze(0)
        with torch.no_grad():
            compressed_data = encoder3d_pool[2](video_tensor)
        data_path = save_compressed_data(video_path, compressed_data)
        print(f'Encoding {video_path} (length={CNN_FRAME_NUM}) to {data_path}')

        file_size = os.path.getsize(data_path)
        file_info = struct.pack(head, os.path.splitext(
            video_name)[0].encode(), file_size, time.time())
        client_socket.send(file_info)
        with open(data_path, "rb") as f:
            client_socket.sendall(f.read())
        print(f'Send {data_path} to server')
        i += 1
client_socket.close()
