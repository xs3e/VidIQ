import os

import torch

from VidIQ.autoencoder.dataset import (IMAGE_DIR, IMAGE_EXT,
                                        load_video_to_tensor)
from VidIQ.autoencoder.util import (IMAGE_EXT, convert_image_to_tensor,
                                     print_device_info)
from VidIQ.dnn_model.faster_rcnn import run_faster_rcnn
from VidIQ.dnn_model.fcos import run_fcos
from VidIQ.dnn_model.keypoint_rcnn import run_keypoint_rcnn
from VidIQ.dnn_model.mask_rcnn import run_mask_rcnn
from VidIQ.dnn_model.ssd import run_ssd
from VidIQ.dnn_model.util import distance_accuracy, f1_score, miou_accuracy
from VidIQ.dnn_model.yolov5 import run_yolov5


def image_inference():
    gt_imgs = [IMAGE_DIR+'/000000013291'+IMAGE_EXT]
    gt_inputs = convert_image_to_tensor(gt_imgs[0], False, True).unsqueeze(0)
    imgs = [IMAGE_DIR+'/000000013291_rec'+IMAGE_EXT]
    inputs = convert_image_to_tensor(imgs[0], False, True).unsqueeze(0)
    run_yolov5(imgs, inputs, gt_imgs, gt_inputs)
    run_ssd(imgs, inputs, gt_imgs, gt_inputs)
    run_faster_rcnn(imgs, inputs, gt_imgs, gt_inputs)
    run_fcos(imgs, inputs, gt_imgs, gt_inputs)
    run_mask_rcnn(imgs, inputs, gt_imgs, gt_inputs)
    run_keypoint_rcnn(imgs, inputs, gt_imgs, gt_inputs)


def prepare_imgs_inputs_for_video(data_path, decoded_tensor, rev):
    base_path = os.path.splitext(data_path)[0]
    video_length = decoded_tensor.shape[2]
    imgs = [os.path.join(base_path+'_rec', 'frame' +
                         "{:04d}".format(_+1)+IMAGE_EXT) for _ in range(video_length)]
    gt_imgs = [os.path.join(
        base_path, 'frame'+"{:04d}".format(_+1)+IMAGE_EXT) for _ in range(video_length)]
    inputs = rev(decoded_tensor)
    gt_inputs, _ = load_video_to_tensor(
        base_path, length=video_length, return_list=True)
    gt_inputs = torch.stack(gt_inputs, dim=0)
    return imgs, inputs, gt_imgs, gt_inputs


def run_dnn_task(dnn_task, imgs, inputs, gt_imgs, gt_inputs, show=False, online=False):
    if dnn_task == "yolov5":
        return run_yolov5(imgs, inputs, gt_imgs, gt_inputs, show, online)
    elif dnn_task == "faster_rcnn":
        return run_faster_rcnn(imgs, inputs, gt_imgs, gt_inputs, show, online)
    elif dnn_task == "fcos":
        return run_fcos(imgs, inputs, gt_imgs, gt_inputs, show, online)
    elif dnn_task == "ssd":
        return run_ssd(imgs, inputs, gt_imgs, gt_inputs, show, online)
    elif dnn_task == "mask_rcnn":
        return run_mask_rcnn(imgs, inputs, gt_imgs, gt_inputs, show, online)
    elif dnn_task == "keypoint_rcnn":
        return run_keypoint_rcnn(imgs, inputs, gt_imgs, gt_inputs, show, online)


def compute_accuracy(dnn_task, results, gt_results):
    if dnn_task == "yolov5":
        return f1_score(results, gt_results, True)
    elif dnn_task == "faster_rcnn" or dnn_task == "fcos" or dnn_task == "ssd":
        return f1_score(results, gt_results)
    elif dnn_task == "mask_rcnn":
        return miou_accuracy(results, gt_results)
    elif dnn_task == "keypoint_rcnn":
        return distance_accuracy(results, gt_results)


if __name__ == '__main__':
    print_device_info()
    image_inference()
