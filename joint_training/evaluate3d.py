import time

import torch

from VidIQ.autoencoder.dataset import (CNN_FRAME_NUM, VIDEO_DIR,
                                        load_video_to_tensor,
                                        show_videos_difference)
from VidIQ.autoencoder.network3d import (get_networks3d,
                                          print_autoencoder3d_info)
from VidIQ.autoencoder.util import (YUV_ENABLED, print_device_info,
                                     save_tensor_to_video)
from VidIQ.super_resolution.models import SCALE_RATIO, get_models

CODEC_MODE = "encoder"
print_device_info()
print_autoencoder3d_info()

# Load trained encoder and decoder
if CODEC_MODE == "encoder":
    encoder3d, _, decoder3d, _ = get_networks3d(
        'eval', True, joint=True, down_factor=SCALE_RATIO)
    model3d, model3d_path = get_models('eval', True, network3d=True)
elif CODEC_MODE == "decoder":
    encoder3d, _ = get_networks3d(
        'eval', True, is_decoder=False, joint=True, down_factor=SCALE_RATIO)
    model3d, model3d_path = get_models(
        'eval', True, integrated=True, network3d=True)

# Encoder compresses test video
video_path = VIDEO_DIR+'/aeroplane_0001_033'
video_tensor = load_video_to_tensor(
    video_path, length=CNN_FRAME_NUM, is_yuv=YUV_ENABLED).unsqueeze(0)
with torch.no_grad():
    compressed_data = encoder3d(video_tensor)

# Video decoding and SR
with torch.no_grad():
    start = time.time()
    if CODEC_MODE == "encoder":
        decoded_tensor = decoder3d(compressed_data)
        sr_tensor = model3d(decoded_tensor)
        print(f'Time for decoder + SR: {time.time()-start}s')
    elif CODEC_MODE == "decoder":
        sr_tensor = model3d(compressed_data)
        print(f'Time for integrated decoder: {time.time()-start}s')

# Save images to video
reconstructed_path = video_path+'_sr'
save_tensor_to_video(reconstructed_path, sr_tensor[0], False)

# Show images in video
show_videos_difference(CNN_FRAME_NUM, video_path, reconstructed_path)
