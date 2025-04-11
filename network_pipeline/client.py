import os
import socket
import struct
import subprocess
import threading
import time

import torch

from VidIQ.autoencoder.dataset import IMAGE_DIR, IMAGE_EXT
from VidIQ.autoencoder.util import (CUDA_ENABLED, PROJECT_DIR, YUV_ENABLED,
                                    convert_image_to_tensor,
                                    save_compressed_data)
from VidIQ.network_pipeline.monolithic_controller import MC
from VidIQ.network_pipeline.util import (default_com_index, head,
                                         load_encoder_pool, samplesize,
                                         server_ip, server_port,
                                         start_http_server, udp_port)

encoder_pool = load_encoder_pool(False)
client_encoder = encoder_pool[default_com_index].to(0)
client_lock = threading.Lock()
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.settimeout(MC.timeout)
client_end = False


def send_sample():
    i = 0
    global client_end
    for file_name in os.listdir(IMAGE_DIR):
        if i >= samplesize:
            break
        if file_name.endswith(IMAGE_EXT):
            file_path = os.path.join(IMAGE_DIR, file_name)
            img_tensor = convert_image_to_tensor(
                file_path, YUV_ENABLED, CUDA_ENABLED).unsqueeze(0)
            with client_lock:
                with torch.no_grad():
                    compressed_data = client_encoder(img_tensor)
            data_path = save_compressed_data(file_path, compressed_data)
            print(f'Encoding {file_path} to {data_path}')

            file = open(data_path, "rb")
            file_content = file.read()
            file_size = len(file_content)
            file_info = struct.pack(head, os.path.splitext(file_name)[
                                    0].encode(), file_size, time.time())
            client_socket.send(file_info)
            client_socket.sendall(file_content)
            file.close()
            print(f'Send {data_path} to server')
            i += 1
    client_socket.close()
    client_end = True


def handle_com_index():
    global default_com_index, client_encoder, client_end
    last_com_index = default_com_index
    switch_num = 0
    switch_overhead = 0
    while not client_end:
        try:
            message, addr = udp_socket.recvfrom(struct.calcsize('i'))
        except socket.timeout:
            break
        if server_ip == addr[0]:
            new_com_index = struct.unpack('i', message)[0]
            if last_com_index != new_com_index:
                with client_lock:
                    t = time.time()
                    client_encoder.cpu()
                    client_encoder = encoder_pool[new_com_index].to(0)
                    switch_overhead += (time.time()-t)
                    switch_num += 1
                print(
                    f"\033[1;32m[DECISION] new_com_index={new_com_index}\033[0m")
                last_com_index = new_com_index
    udp_socket.close()
    if switch_num != 0:
        print(
            f"\033[1;32m[CLIENT] switch_overhead={switch_overhead/switch_num}s\033[0m")


if __name__ == "__main__":
    sender = threading.Thread(target=send_sample)
    handler = threading.Thread(target=handle_com_index)
    gt_server = start_http_server()
    client_socket.connect((server_ip, server_port))
    udp_ip = subprocess.check_output(
        ['bash', PROJECT_DIR+'/network_pipeline/get_ip.sh']).decode().strip()
    print(f'UDP IP is {udp_ip}')
    udp_socket.bind((udp_ip, udp_port))
    handler.start()
    sender.start()
    sender.join()
    print("\033[1;32m[Sender exits]\033[0m")
    handler.join()
    print("\033[1;32m[Handler exits]\033[0m")
    gt_server.shutdown()
    gt_server.server_close()
    time.sleep(MC.update_time_slot)
    print("\033[1;32m[GTServer exits]\033[0m")
