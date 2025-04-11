import os

import torch

from VidIQ.autoencoder.dataset import (CNN_FRAME_NUM, VIDEO_DIR,
                                        load_video_to_tensor,
                                        show_videos_difference)
from VidIQ.autoencoder.network3d import (get_networks3d,
                                          print_autoencoder3d_info)
from VidIQ.autoencoder.util import (YUV_ENABLED, get_folder_size,
                                     load_compressed_data, print_device_info,
                                     save_compressed_data,
                                     save_tensor_to_video)

print_device_info()
print_autoencoder3d_info()

# Load trained encoder and decoder
encoder3d, _, decoder3d, _ = get_networks3d('eval', True)

# Encoder compresses test video
frame_num = CNN_FRAME_NUM
video_path = VIDEO_DIR+'/aeroplane_0001_033'
video_size = get_folder_size(video_path)
print(f"Size of original video {video_path} is {video_size:.4f} KB")
video_tensor = load_video_to_tensor(
    video_path, length=frame_num, is_yuv=YUV_ENABLED).unsqueeze(0)
with torch.no_grad():
    compressed_data = encoder3d(video_tensor)

# Save compressed data
data_path = save_compressed_data(video_path, compressed_data)
data_size = os.path.getsize(data_path)/1024
print(f"Size of compressed data {data_path} is {data_size:.4f} KB")

# Load compressed data
compressed_data = load_compressed_data(data_path)

# Extract compressed data to images
with torch.no_grad():
    decoded_tensor = decoder3d(compressed_data)

# Save images to video
reconstructed_path = video_path+'_rec'
save_tensor_to_video(reconstructed_path, decoded_tensor[0], False)
reconstructed_size = get_folder_size(reconstructed_path)
print(
    f"Size of reconstructed video {reconstructed_path} is {reconstructed_size:.4f} KB")

# Show images in video
show_videos_difference(frame_num, video_path, reconstructed_path)
