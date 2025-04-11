import torch

from VidIQ.autoencoder.dataset import IMAGE_DIR, IMAGE_EXT
from VidIQ.autoencoder.util import (IMAGE_EXT, YUV_ENABLED,
                                     convert_image_to_tensor,
                                     convert_tensor_to_image,
                                     print_device_info)
from VidIQ.dnn_model.faster_rcnn import run_faster_rcnn
from VidIQ.dnn_model.fcos import run_fcos
from VidIQ.dnn_model.yolov5 import run_yolov5
from VidIQ.super_resolution.models import LR_IMAGE_DIR, get_models


def image_sr_inference():
    name = '/000000013291'
    model, _ = get_models('eval', True)
    gt_imgs = [IMAGE_DIR+name+IMAGE_EXT]
    gt_inputs = convert_image_to_tensor(gt_imgs[0], False, True).unsqueeze(0)
    imgs = [LR_IMAGE_DIR+name+IMAGE_EXT]
    inputs = convert_image_to_tensor(imgs[0], False, True).unsqueeze(0)
    run_yolov5(imgs, inputs, gt_imgs, gt_inputs)
    run_faster_rcnn(imgs, inputs, gt_imgs, gt_inputs)
    run_fcos(imgs, inputs, gt_imgs, gt_inputs)
    # *image super resolution
    inputs = convert_image_to_tensor(imgs[0], YUV_ENABLED, True).unsqueeze(0)
    with torch.no_grad():
        sr_tensor = model(inputs)
    sr_path = LR_IMAGE_DIR+name+'_sr'+IMAGE_EXT
    sr_img = convert_tensor_to_image(sr_tensor[0], False)
    sr_img.save(sr_path)
    run_yolov5([sr_path], sr_tensor, gt_imgs, gt_inputs)
    run_faster_rcnn([sr_path], sr_tensor, gt_imgs, gt_inputs)
    run_fcos([sr_path], sr_tensor, gt_imgs, gt_inputs)


if __name__ == '__main__':
    print_device_info()
    image_sr_inference()
