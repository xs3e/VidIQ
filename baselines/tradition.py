import os
import shutil
import time

import imageio
import torch
import torchvision.transforms as transforms
from PIL import Image

from VidIQ.autoencoder.dataset import (IMAGE_DIR, IMAGE_EXT, IMAGE_HEIGHT,
                                        IMAGE_WIDTH, VIDEO_DIR)
from VidIQ.autoencoder.util import IMAGE_EXT
from VidIQ.super_resolution.models import SCALE_RATIO


def pil_image_to_tensor(pil_image, is_yuv=False):
    if is_yuv:
        pil_image = pil_image.convert('YCbCr')
    tensor = transforms.ToTensor()(pil_image).unsqueeze(0).to(0)
    return tensor


def image_jpeg(image_path, qp):
    image = Image.open(image_path + IMAGE_EXT)
    jpeg_path = image_path+".jpeg"
    image.save(jpeg_path, "JPEG", quality=qp)
    image = Image.open(jpeg_path)
    tensor = pil_image_to_tensor(image)
    kbyte = os.path.getsize(jpeg_path)/1024
    return tensor, kbyte


def load_video(video_directory, frame_num=0, is_yuv=False):
    if frame_num == 0:
        frames = [os.path.splitext(img)[0] for img in os.listdir(
            video_directory) if img.endswith(IMAGE_EXT)]
        frames.sort()
        frame_num = len(frames)
    else:
        frames = ['frame'+"{:04d}".format(_+1) for _ in range(frame_num)]
    video = []
    for _ in range(frame_num):
        image_path = os.path.join(video_directory, frames[_])
        image = Image.open(image_path + IMAGE_EXT)
        if is_yuv:
            image = image.convert('YCbCr')
        video.append(image)
    return video, frame_num


def save_low_resolution_video(video_directory, scale_ratio):
    lr_video_directory = video_directory+'_lrx'+str(scale_ratio)
    if os.path.exists(lr_video_directory):
        shutil.rmtree(lr_video_directory)
    os.mkdir(lr_video_directory)
    video, frame_num = load_video(video_directory)
    for _ in range(frame_num):
        resized_img = video[_].resize(
            (IMAGE_WIDTH//scale_ratio, IMAGE_HEIGHT//scale_ratio), Image.BICUBIC)
        resized_img.save(lr_video_directory+'/frame' +
                         "{:04d}".format(_+1) + IMAGE_EXT)
    return lr_video_directory


def encode_images_to_mp4(video_directory, qp, frame_num=0, is_yuv=False):
    t = time.time()
    video, frame_num = load_video(video_directory, frame_num, is_yuv)
    video_writer = imageio.v2.get_writer(
        video_directory+'.mp4', fps=30, ffmpeg_params=['-c:v', 'hevc_nvenc', '-qp', str(qp)])
    for _ in range(frame_num):
        img_array = imageio.core.asarray(video[_])
        video_writer.append_data(img_array)
    video_writer.close()
    print(f"Video encoding time:{time.time()-t} s")
    kbyte = os.path.getsize(video_directory+'.mp4')/1024
    return kbyte


def decode_mp4_to_images(video_mp4, is_yuv=False, return_list=True):
    reader = imageio.get_reader(
        video_mp4, ffmpeg_params=['-r', '30', '-c:v', 'hevc_cuvid'])
    image_tensors = []
    for frame in reader:
        pil_image = Image.fromarray(frame)
        tensor = pil_image_to_tensor(pil_image, is_yuv)
        if return_list:
            image_tensors.append(tensor)
        else:
            image_tensors.append(tensor[0])
    if return_list:
        return image_tensors
    else:
        return torch.stack(image_tensors, dim=1).unsqueeze(0)


if __name__ == "__main__":
    image_path = IMAGE_DIR+'/000000013291'
    tensor, kbyte = image_jpeg(image_path, 20)
    video_directory = VIDEO_DIR+'/dog_0014_021'
    kbyte = encode_images_to_mp4(video_directory, 51)
    kbyte = encode_images_to_mp4(video_directory, 51, True)
    image_tensors = decode_mp4_to_images(video_directory+'.mp4')
    save_low_resolution_video(video_directory, SCALE_RATIO)
