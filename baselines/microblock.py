import os
import random
import time
from io import BytesIO

import imageio
import torch
from PIL import Image
from torchvision.ops import box_iou

from VidIQ.autoencoder.dataset import (IMAGE_DIR, IMAGE_EXT, IMAGE_HEIGHT,
                                        IMAGE_WIDTH, VIDEO_DIR)
from VidIQ.autoencoder.loss import loss_function
from VidIQ.autoencoder.util import IMAGE_EXT, YUV_ENABLED
from VidIQ.baselines.tradition import (decode_mp4_to_images, load_video,
                                        pil_image_to_tensor)
from VidIQ.dnn_model.util import f1_score, get_boxes_and_labels
from VidIQ.dnn_model.yolov5 import load_yolov5
from VidIQ.super_resolution.models import (LR_IMAGE_DIR, SCALE_RATIO,
                                            get_models)

BLOCK_SIZE = 64
PSNR = loss_function(2)
model, model_path = get_models('eval', True)


def get_block_size(width, height):
    if width == IMAGE_WIDTH and height == IMAGE_HEIGHT:
        block_size = BLOCK_SIZE
    else:
        block_size = BLOCK_SIZE//SCALE_RATIO
    if width % block_size != 0 or height % block_size != 0:
        print(
            f"Warning: block size {block_size} is not fit for {width}*{height} image")
    return block_size


def get_num_blocks_hw(image):
    width, height = image.size
    block_size = get_block_size(width, height)
    num_blocks_height = height // block_size
    num_blocks_width = width // block_size
    return block_size, num_blocks_height, num_blocks_width


def generate_coding_level(num_blocks_height, num_blocks_width, value=None):
    if value is not None:
        return [[value for _ in range(num_blocks_width)] for _ in range(num_blocks_height)]
    else:
        return [[random.randint(0, 100) for _ in range(num_blocks_width)] for _ in range(num_blocks_height)]


def crop_image(block_size, image, i, j):
    upper = i * block_size
    left = j * block_size
    lower = upper + block_size
    right = left + block_size
    block = image.crop((left, upper, right, lower))
    return block, left, upper


def image_microblock_coding(image, image_path, coding_level=None):
    t = time.time()
    block_size, num_blocks_height, num_blocks_width = get_num_blocks_hw(image)
    if coding_level:
        assert num_blocks_height == len(coding_level)
        assert num_blocks_width == len(coding_level[0])
    else:
        coding_level = generate_coding_level(
            num_blocks_height, num_blocks_width)
        # print(coding_level)
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            block, left, upper = crop_image(block_size, image, i, j)
            output = BytesIO()
            block.save(output, "JPEG", quality=coding_level[i][j])
            block = Image.open(output)
            image.paste(block, (left, upper))
    if image_path:
        image.save(image_path+".jpeg")
        print(f"Image encoding time:{time.time()-t} s")
        kbyte = os.path.getsize(image_path+".jpeg")/1024
        return image, kbyte
    return image


def profiling_block_qp_psnr(lr_block, block):
    lr_tensor = pil_image_to_tensor(lr_block, YUV_ENABLED)
    tensor = pil_image_to_tensor(block)
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    gt_loss = PSNR(sr_tensor, tensor).item()
    result_qp = 100
    step = 10
    for qp in range(0, 100, step):
        output = BytesIO()
        lr_block.save(output, "JPEG", quality=qp)
        lr_tensor = pil_image_to_tensor(Image.open(output), YUV_ENABLED)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        loss = PSNR(sr_tensor, tensor).item()
        if abs(loss - gt_loss)/gt_loss <= 1:
            result_qp = qp
            break
    return result_qp


def AccelIR(lr_image_path, image_path):
    lr_image = Image.open(lr_image_path + IMAGE_EXT)
    image = Image.open(image_path + IMAGE_EXT)
    block_size, num_blocks_height, num_blocks_width = get_num_blocks_hw(image)
    coding_level = generate_coding_level(
        num_blocks_height, num_blocks_width, 0)
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            lr_block, _, _ = crop_image(
                block_size//SCALE_RATIO, lr_image, i, j)
            block, _, _ = crop_image(block_size, image, i, j)
            coding_level[i][j] = profiling_block_qp_psnr(lr_block, block)
    lr_image, kbyte = image_microblock_coding(
        lr_image, lr_image_path, coding_level)
    return lr_image, kbyte


def profiling_block_qp_acc(image, dnn, nms_module=None, low_qp=0):
    block_size, num_blocks_height, num_blocks_width = get_num_blocks_hw(image)
    coding_level = generate_coding_level(
        num_blocks_height, num_blocks_width, low_qp)
    tensor = pil_image_to_tensor(image)
    with torch.no_grad():
        gt_results = dnn(tensor)
    is_yolov5 = False
    if nms_module is not None:
        gt_results = nms_module.non_max_suppression(gt_results)
        is_yolov5 = True
    boxes, _ = get_boxes_and_labels(gt_results, 0, is_yolov5)
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            block = torch.tensor(
                [[j*block_size, i*block_size, (j+1)*block_size, (i+1)*block_size]]).to(0)
            iou = box_iou(block, boxes)
            if not torch.all(iou == 0):
                coding_level[i][j] = 100
    return gt_results, coding_level


def AccMPEG_image(image_path, dnn, nms_module=None, is_yuv=False, low_qp=0):
    image = Image.open(image_path + IMAGE_EXT)
    gt_results, coding_level = profiling_block_qp_acc(
        image, dnn, nms_module, low_qp)
    image, kbyte = image_microblock_coding(image, image_path, coding_level)
    tensor = pil_image_to_tensor(image, is_yuv)
    return gt_results, tensor, kbyte


def video_microblock_coding(video_directory, coding_levels=None):
    t = time.time()
    video, frame_num = load_video(video_directory)
    if coding_levels:
        assert len(coding_levels) == frame_num
    video_writer = imageio.v2.get_writer(
        video_directory+'.mp4', fps=30, ffmpeg_params=['-c:v', 'hevc_nvenc', '-qp', '0'])
    for _ in range(frame_num):
        if coding_levels:
            coding_level = coding_levels[_]
        else:
            coding_level = None
        image = image_microblock_coding(video[_], None, coding_level)
        img_array = imageio.core.asarray(image)
        video_writer.append_data(img_array)
    video_writer.close()
    print(f"Video encoding time:{time.time()-t} s")
    kbyte = os.path.getsize(video_directory+'.mp4')/1024
    return kbyte


def AccMPEG_video(video_directory, dnn, nms_module=None):
    video, frame_num = load_video(video_directory)
    video_gt_results = []
    coding_levels = []
    for _ in range(frame_num):
        gt_results, coding_level = profiling_block_qp_acc(
            video[_], dnn, nms_module)
        video_gt_results.append(gt_results)
        coding_levels.append(coding_level)
    kbyte = video_microblock_coding(video_directory, coding_levels)
    return video_gt_results, frame_num, kbyte


if __name__ == "__main__":
    lr_image_path = LR_IMAGE_DIR+'/000000013291'
    image_path = IMAGE_DIR+'/000000013291'
    yolov5, nms_module, _, _ = load_yolov5()

    gt_results, tensor, kbyte = AccMPEG_image(image_path, yolov5, nms_module)
    with torch.no_grad():
        results = yolov5(tensor)
    results = nms_module.non_max_suppression(results)
    acc = f1_score(results, gt_results, True)
    print(f"AccMPEG image accuracy is {acc}")

    lr_image, kbyte = AccelIR(lr_image_path, image_path)
    lr_tensor = pil_image_to_tensor(lr_image, YUV_ENABLED)
    with torch.no_grad():
        results = yolov5(model(lr_tensor))
    results = nms_module.non_max_suppression(results)
    acc = f1_score(results, gt_results, True)
    print(f"AccelIR image accuracy is {acc}")

    video_directory = VIDEO_DIR+'/dog_0014_021'
    video_gt_results, frame_num, kbyte = AccMPEG_video(
        video_directory, yolov5, nms_module)
    image_tensors = decode_mp4_to_images(video_directory+".mp4")
    assert len(image_tensors) == frame_num
    acc = 0
    for _ in range(frame_num):
        with torch.no_grad():
            results = yolov5(image_tensors[_])
        results = nms_module.non_max_suppression(results)
        acc += f1_score(results, video_gt_results[_], True)
    acc /= frame_num
    print(f"AccMPEG video accuracy is {acc}")
