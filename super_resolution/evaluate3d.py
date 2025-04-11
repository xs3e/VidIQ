import torch

from VidIQ.autoencoder.dataset import (CNN_FRAME_NUM, IMAGE_HEIGHT,
                                        IMAGE_WIDTH, VIDEO_DIR,
                                        load_video_to_tensor,
                                        show_videos_difference)
from VidIQ.autoencoder.util import (YUV_ENABLED, get_folder_size,
                                     print_device_info, save_tensor_to_video)
from VidIQ.super_resolution.models import SCALE_RATIO, get_models

print_device_info()
model3d, model3d_path = get_models('eval', True, network3d=True)

# Encoder compresses test video
frame_num = CNN_FRAME_NUM
video_path = VIDEO_DIR+'/aeroplane_0001_033'
video_size = get_folder_size(video_path)
print(f"Size of original video {video_path} is {video_size:.4f} KB")
video_tensor = load_video_to_tensor(
    video_path, length=frame_num, is_yuv=YUV_ENABLED,
    resize=[IMAGE_HEIGHT//SCALE_RATIO, IMAGE_WIDTH//SCALE_RATIO]).unsqueeze(0)
with torch.no_grad():
    sr_video_tensor = model3d(video_tensor)

# Save images to video
reconstructed_path = video_path+'_sr'
save_tensor_to_video(reconstructed_path, sr_video_tensor[0], False)

# Show images in video
show_videos_difference(frame_num, video_path, reconstructed_path)
