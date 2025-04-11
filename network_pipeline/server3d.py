import socket
import threading

import torch

from VidIQ.autoencoder.dataset import VIDEO_DIR
from VidIQ.autoencoder.util import load_compressed_data, save_tensor_to_video
from VidIQ.dnn_model.inference import (prepare_imgs_inputs_for_video,
                                       run_dnn_task)
from VidIQ.network_pipeline.monolithic_controller import MonolithicController
from VidIQ.network_pipeline.util import (client_bandwidth, dnn_task,
                                         load_decoder_pool, load_sr_model_pool,
                                         result_queue, rev, sample_queue,
                                         samplesize, server_ip, server_port,
                                         tcp_recv)

decoder3d_pool = load_decoder_pool()
sr_model3d_pool = load_sr_model_pool()
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def receive_sample():
    i = 0
    while i < samplesize:
        base_path, data_path, curr_bandwidth = tcp_recv(conn, VIDEO_DIR)
        print(f'Received {data_path} from client ({curr_bandwidth} Mbps)')
        sample_queue.put(base_path)
        if not client_bandwidth.empty():
            _ = client_bandwidth.get()
        client_bandwidth.put(curr_bandwidth)
        i += 1
    conn.close()
    server_socket.close()


def process_sample():
    i = 0
    while i < samplesize:
        base_path = sample_queue.get()
        data_path = base_path+'.pkl'
        compressed_data = load_compressed_data(data_path)
        with torch.no_grad():
            decoded_tensor = decoder3d_pool[2](compressed_data)
        reconstructed_path = base_path+'_rec'
        save_tensor_to_video(reconstructed_path, decoded_tensor[0], False)
        print(f'Decoding {data_path} to {reconstructed_path}')
        imgs, inputs, gt_imgs, gt_inputs = prepare_imgs_inputs_for_video(
            data_path, decoded_tensor, rev)
        result_queue.put(run_dnn_task(
            dnn_task, imgs, inputs, gt_imgs, gt_inputs))
        print(f'Analyzed {reconstructed_path}')
        i += 1


if __name__ == "__main__":
    receiver = threading.Thread(target=receive_sample)
    processer = threading.Thread(target=process_sample)
    mc = MonolithicController(0.85, 5, 4, 8)
    controller = threading.Thread(target=mc.online_decisions)
    server_socket.bind((server_ip, server_port))
    server_socket.settimeout(20)
    try:
        server_socket.listen(1)
        print('Wait for client to connect...')
        conn, addr = server_socket.accept()
        print('Client is connected:', addr)
    except socket.timeout:
        print('No connection received within 20s, receiver exits')
        server_socket.close()
        exit(0)
    controller.start()
    processer.start()
    receiver.start()
    receiver.join()
    processer.join()
    controller.join()
