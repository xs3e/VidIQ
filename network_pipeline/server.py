import queue
import socket
import threading
import time

import torch

from VidIQ.autoencoder.dataset import IMAGE_DIR, IMAGE_EXT
from VidIQ.autoencoder.util import load_compressed_data
from VidIQ.dnn_model.inference import run_dnn_task
from VidIQ.network_pipeline.monolithic_controller import MC
from VidIQ.network_pipeline.util import (client_bandwidth, default_com_index,
                                         default_sr_index, dnn_task,
                                         load_decoder_pool, load_sr_model_pool,
                                         new_server_decoder,
                                         old_server_decoder, result_queue,
                                         sample_queue, samplesize,
                                         server_com_index, server_ip,
                                         server_port, server_sr_index,
                                         tcp_recv)

decoder_pool = load_decoder_pool(False)
server_decoder = decoder_pool[default_com_index].to(0)
sr_model_pool = load_sr_model_pool(False)
server_sr_model = sr_model_pool[default_sr_index].to(0)
new_com_index = 0
new_sr_index = 0
server_lock = threading.Lock()
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_end = False


def receive_sample():
    i = 0
    while i < samplesize:
        base_path, data_path, bandwidth = tcp_recv(conn, IMAGE_DIR)
        print(f'Received {data_path} from client ({bandwidth} Mbps)')
        sample_queue.put(base_path)
        if client_bandwidth.full():
            _ = client_bandwidth.get()
        client_bandwidth.put(bandwidth)
        i += 1
    conn.close()
    server_socket.close()


def process_sample():
    i = 0
    global server_decoder, server_sr_model, server_end
    global old_server_decoder, new_server_decoder
    last_shape = None
    while i < samplesize:
        base_path = sample_queue.get()
        data_path = base_path+'.pkl'
        compressed_data = load_compressed_data(data_path)
        if last_shape is not None:
            if compressed_data.shape != last_shape:
                try:
                    decoder = server_decoder
                    server_decoder = new_server_decoder.get(timeout=MC.timeout)
                    old_server_decoder.put(decoder, timeout=MC.timeout)
                except queue.Empty or queue.Full:
                    break
        last_shape = compressed_data.shape
        with torch.no_grad():
            inputs = server_decoder(compressed_data)
        res = run_dnn_task(dnn_task, None, inputs, None, None, online=True)
        item = [base_path, new_com_index,
                new_sr_index, MC.online_bandwidth, res]
        result_queue.put(item)
        print(f'Analyzed {base_path}_rec{IMAGE_EXT}')
        i += 1
    server_end = True


def handle_com_index_and_sr_index():
    global new_com_index, new_sr_index, server_end
    global server_sr_model, old_server_decoder, new_server_decoder
    global default_com_index, default_sr_index
    last_com_index = default_com_index
    last_sr_index = default_sr_index
    switch_num = 0
    switch_overhead = 0
    while not server_end:
        try:
            new_com_index = server_com_index.get(timeout=MC.decision_time_slot)
            new_sr_index = server_sr_index.get(timeout=MC.decision_time_slot)
            if last_com_index != new_com_index:
                t = time.time()
                decoder = decoder_pool[new_com_index].to(0)
                switch_overhead += (time.time()-t)
                switch_num += 1
                new_server_decoder.put(decoder, timeout=MC.timeout)
                decoder = old_server_decoder.get(timeout=MC.timeout)
                decoder.cpu()
                last_com_index = new_com_index
        except queue.Empty or queue.Full:
            break
        if last_sr_index != new_sr_index:
            with server_lock:
                if server_sr_model:
                    server_sr_model.cpu()
                sr_model = sr_model_pool[new_sr_index]
                if sr_model:
                    server_sr_model = sr_model.to(0)
            last_sr_index = new_sr_index
    if switch_num != 0:
        print(
            f"\033[1;32m[SERVER] switch_overhead={switch_overhead/switch_num}s\033[0m")


if __name__ == "__main__":
    receiver = threading.Thread(target=receive_sample)
    processer = threading.Thread(target=process_sample)
    controller = threading.Thread(target=MC.online_decisions)
    updater = threading.Thread(target=MC.online_update)
    handler = threading.Thread(target=handle_com_index_and_sr_index)
    server_socket.bind((server_ip, server_port))
    server_socket.settimeout(20)
    try:
        server_socket.listen(1)
        print('Wait for client to connect...')
        conn, addr = server_socket.accept()
        MC.client_address = addr[0]
        print('Client is connected:', addr)
    except socket.timeout:
        print('No connection received within 20s, receiver exits')
        server_socket.close()
        exit(0)
    handler.start()
    updater.start()
    controller.start()
    processer.start()
    receiver.start()
    receiver.join()
    print("\033[1;32m[Receiver exits]\033[0m")
    processer.join()
    print("\033[1;32m[Processer exits]\033[0m")
    controller.join()
    print("\033[1;32m[Controller exits]\033[0m")
    updater.join()
    print("\033[1;32m[Updater exits]\033[0m")
    handler.join()
    print("\033[1;32m[Handler exits]\033[0m")
