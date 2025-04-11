import time

import matplotlib.pyplot as plt
import torch

from VidIQ.autoencoder.network2d import get_networks, print_autoencoder_info
from VidIQ.autoencoder.util import (CUDA_ENABLED, IMAGE_EXT, YUV_ENABLED,
                                     convert_image_to_tensor,
                                     convert_tensor_to_image,
                                     print_device_info)
from VidIQ.super_resolution.models import (IMAGE_DIR, LR_IMAGE_DIR,
                                            SCALE_RATIO, get_models)

CODEC_MODE = "encoder"
print_device_info()
print_autoencoder_info()

# Load trained encoder and decoder
if CODEC_MODE == "encoder":
    encoder, _, decoder, _ = get_networks(
        'eval', True, joint=True, down_factor=SCALE_RATIO)
    model, model_path = get_models('eval', True)
elif CODEC_MODE == "decoder":
    encoder, _ = get_networks(
        'eval', True, is_decoder=False, joint=True, down_factor=SCALE_RATIO)
    model, model_path = get_models('eval', True, integrated=True)

# Encoder compresses test image
name = '/000000013291'
img_path = IMAGE_DIR+name+IMAGE_EXT
img_tensor = convert_image_to_tensor(
    img_path, YUV_ENABLED, CUDA_ENABLED).unsqueeze(0)
with torch.no_grad():
    compressed_data = encoder(img_tensor)

# Image decoding and SR
with torch.no_grad():
    start = time.time()
    if CODEC_MODE == "encoder":
        decoded_tensor = decoder(compressed_data)
        sr_tensor = model(decoded_tensor)
        print(f'Time for decoder + SR: {time.time()-start}s')
    elif CODEC_MODE == "decoder":
        sr_tensor = model(compressed_data)
        print(f'Time for integrated decoder: {time.time()-start}s')

# Save image
sr_img = convert_tensor_to_image(sr_tensor[0], False)
sr_path = LR_IMAGE_DIR+name+'_sr'+IMAGE_EXT
sr_img.save(sr_path)

# Load image
input_img = plt.imread(img_path)
sr_img = plt.imread(sr_path)

# Show image
plt.subplot(1, 2, 1)
plt.imshow(input_img)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(sr_img)
plt.title('SR Image')
plt.show()
