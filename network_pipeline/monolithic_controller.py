import queue
import random
import socket
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from VidIQ.autoencoder.dataset import VIDEO_DIR
from VidIQ.autoencoder.util import DOWNLOAD_DIR
from VidIQ.dnn_model.inference import compute_accuracy, run_dnn_task
from VidIQ.network_pipeline.util import (client_bandwidth, com_factor_num,
                                         dnn_task,
                                         fetch_images_from_http_server,
                                         get_fit_video_paths,
                                         get_network_depth, get_scale_ratio,
                                         pipeline_benchmarks, result_num,
                                         result_queue, server_com_index,
                                         server_sr_index, sr_multiplier_num,
                                         udp_port)


def func(x, a, b, c):
    return np.log(a*x+b) + c


def fitting_accuracy(x_data, y_data, initial_params, show=False):
    params, _ = curve_fit(func, x_data, y_data, p0=initial_params)
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    if show:
        y_fit = func(x_fit, *params)
        plt.scatter(x_data, y_data, label='Data')
        plt.plot(x_fit, y_fit, label='Fit', color='red')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    return params


def get_com_factor(com_index):
    return pow(2, get_network_depth(com_index))


def compression_profiling(com_factor, com_fitting_params, online_com_index=None,
                          old_acc=None, online_acc=None, show=False):
    com_acc = [func(com_factor[_], *com_fitting_params)
               for _ in range(com_factor_num)]
    if online_acc is None:
        # offline
        assert (online_com_index is None) and (old_acc is None)
        # video_paths = get_fit_video_paths()
        # for _ in range(com_factor_num):
        #     com_acc[_] = pipeline_benchmarks(video_paths, _)
        com_acc = [0.91, 0.88, 0.72, 0.48]
    else:
        # online
        assert (online_com_index is not None) and (old_acc is not None)
        assert online_com_index >= 0 and online_com_index < com_factor_num
        com_acc[online_com_index] *= (online_acc/old_acc)
    com_fitting_params = fitting_accuracy(
        com_factor, com_acc, com_fitting_params, show)
    print(com_fitting_params)
    return com_fitting_params


def enhancement_profiling(sr_multiplier, sr_fitting_params, online_com_index=None, online_sr_index=None,
                          online_acc=None, show=False):
    sr_acc = [[func(sr_multiplier[j], *(sr_fitting_params[i]))
               for j in range(sr_multiplier_num)] for i in range(com_factor_num)]
    if online_acc is None:
        # offline
        assert (online_com_index is None) and (online_sr_index is None)
        sr_acc = [[0.72, 0.86, 0.89, 0.99],
                  [0.69, 0.82, 0.86, 0.93],
                  [0.68, 0.73, 0.83, 0.87],
                  [0.60, 0.72, 0.77, 0.81]]
        # video_paths = get_fit_video_paths()
        for i in range(com_factor_num):
            for j in range(sr_multiplier_num):
                pass
                # sr_acc[i][j] = pipeline_benchmarks(
                #     video_paths, i, sr_multiplier_num, sr_index=j)
            params = fitting_accuracy(
                sr_multiplier, sr_acc[i], sr_fitting_params[i], show)
            sr_fitting_params[i] = params
    else:
        # online
        assert (online_com_index is not None) and (online_sr_index is not None)
        assert online_com_index >= 0 and online_com_index < com_factor_num
        assert online_sr_index >= 0 and online_sr_index < sr_multiplier_num
        sr_acc[online_com_index][online_sr_index] = online_acc
        params = fitting_accuracy(
            sr_multiplier, sr_acc[online_com_index],
            sr_fitting_params[online_com_index], show)
        ratio_a = params[0]/sr_fitting_params[online_com_index][0]
        ratio_b = params[1]/sr_fitting_params[online_com_index][1]
        ratio_c = params[2]/sr_fitting_params[online_com_index][2]
        for _ in range(com_factor_num):
            if _ == online_com_index:
                sr_fitting_params[_] = params
            else:
                sr_fitting_params[_][0] *= ratio_a
                sr_fitting_params[_][1] *= ratio_b
                sr_fitting_params[_][2] *= ratio_c
    print(sr_fitting_params)
    return sr_fitting_params


class MonolithicController():
    def __init__(self, base_accuracy, base_bandwidth,
                 decision_time_slot, update_time_slot):
        self.base_accuracy = base_accuracy
        self.base_bandwidth = base_bandwidth
        self.online_bandwidth = base_bandwidth
        self.fluctuation = 0.5
        self.decision_time_slot = decision_time_slot
        self.timeout = 2*decision_time_slot
        self.update_time_slot = update_time_slot
        self.com_fitting_params = [1, 1, 1]
        self.sr_fitting_params = [[1, 1, 1] for _ in range(com_factor_num)]
        self.com_factor = [get_com_factor(_) for _ in range(com_factor_num)]
        self.com_fitting_params = compression_profiling(
            self.com_factor, self.com_fitting_params)
        self.sr_multiplier = [get_scale_ratio(
            _) for _ in range(sr_multiplier_num)]
        self.sr_fitting_params = enhancement_profiling(
            self.sr_multiplier, self.sr_fitting_params)
        print(f"Offline fitting is finished")
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_address = None

    def make_decisions(self, online_bandwidth):
        new_com_index = 0
        for _ in range(com_factor_num-1, -1, -1):
            acc = func(self.com_factor[_], *self.com_fitting_params)
            if acc >= self.base_accuracy:
                new_com_index = _
                break
        com_acc = func(
            self.com_factor[new_com_index], *self.com_fitting_params)
        new_sr_index = sr_multiplier_num-1
        for _ in range(sr_multiplier_num):
            acc = func(self.sr_multiplier[_], *
                       (self.sr_fitting_params[new_com_index]))
            if acc >= self.base_accuracy:
                new_sr_index = _
                break
        base_bandwidth_down = self.base_bandwidth*(1-self.fluctuation)
        base_bandwidth_up = self.base_bandwidth*(1+self.fluctuation)
        if online_bandwidth < base_bandwidth_down:
            new_com_index += 1
            if new_com_index >= com_factor_num:
                new_com_index = com_factor_num-1
            if random.random() > com_acc:
                new_sr_index += 1
                if new_sr_index >= sr_multiplier_num:
                    new_sr_index = sr_multiplier_num-1
        elif online_bandwidth > base_bandwidth_up:
            new_com_index -= 1
            if new_com_index < 0:
                new_com_index = 0
            if random.random() < com_acc:
                new_sr_index -= 1
                if new_sr_index < 0:
                    new_sr_index = 0
        return new_com_index, new_sr_index

    def update_params(self, online_com_index, online_sr_index, online_bandwidth, online_acc):
        old_acc = func(
            self.sr_multiplier[online_sr_index], *(self.sr_fitting_params[online_com_index]))
        self.com_fitting_params = compression_profiling(
            self.com_factor, self.com_fitting_params, online_com_index, old_acc, online_acc)
        self.sr_fitting_params = enhancement_profiling(
            self.sr_multiplier, self.sr_fitting_params, online_com_index, online_sr_index, online_acc)
        self.base_bandwidth = (self.base_accuracy/online_acc)*online_bandwidth

    def online_decisions(self):
        while True:
            try:
                self.online_bandwidth = client_bandwidth.get(
                    timeout=self.decision_time_slot)
            except queue.Empty:
                self.udp_socket.close()
                break
            new_com_index, new_sr_index = self.make_decisions(
                self.online_bandwidth)
            server_com_index.put(new_com_index)
            server_sr_index.put(new_sr_index)
            print(
                f"\033[1;32m[DECISION] new_com_index={new_com_index} new_sr_index={new_sr_index} ({self.online_bandwidth} Mbps)\033[0m")
            time.sleep(self.decision_time_slot)
            self.udp_socket.sendto(struct.pack(
                'i', new_com_index), (self.client_address, udp_port))
        self.udp_socket.close()
        self.client_address = None

    def online_update(self):
        while self.client_address:
            time.sleep(self.update_time_slot)
            paths = []
            results = []
            try:
                for _ in range(result_num):
                    base_path, com_index, sr_index, bandwidth, res = result_queue.get(
                        timeout=self.update_time_slot)
                    if VIDEO_DIR in base_path:
                        base_path += 'frame0001'
                    paths.append(base_path.replace(DOWNLOAD_DIR, ''))
                    results.append(res[0])
            except queue.Empty:
                break
            http_res = fetch_images_from_http_server(
                self.client_address, paths)
            if http_res is None:
                break
            t = time.time()
            gt_results = run_dnn_task(
                dnn_task, None, http_res[0], None, None, online=True)
            online_acc = compute_accuracy(dnn_task, results, gt_results)
            self.update_params(com_index, sr_index, bandwidth, online_acc)
            update_time = time.time()-t
            print(f"Online update time is {update_time} s")
            print(
                f"\033[1;32m[UPDATING] online_com_index={com_index} online_sr_index={sr_index} ({bandwidth} Mbps)\033[0m")


MC = MonolithicController(0.85, 5, 4, 8)

if __name__ == "__main__":
    mc = MonolithicController(0.85, 2, 4, 8)
    new_com_index, new_sr_index = mc.make_decisions(4)
    print(new_com_index, new_sr_index)
    online_acc = np.mean([0.83, 0.73, 0.80, 0.79])
    mc.update_params(new_com_index, new_sr_index, 4, online_acc)
