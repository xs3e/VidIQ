import matplotlib.pyplot as plt
import torch

from VidIQ.autoencoder.util import (CUDA_ENABLED, IMAGE_EXT, YUV_ENABLED,
                                     convert_image_to_tensor,
                                     convert_tensor_to_image,
                                     print_device_info)
from VidIQ.super_resolution.models import LR_IMAGE_DIR, get_models

print_device_info()
model, model_path = get_models('eval', True)

name = '/000000013291'
img_path = LR_IMAGE_DIR+name+IMAGE_EXT

# SR image
img_tensor = convert_image_to_tensor(
    img_path, YUV_ENABLED, CUDA_ENABLED).unsqueeze(0)
with torch.no_grad():
    sr_tensor = model(img_tensor)
sr_img = convert_tensor_to_image(sr_tensor[0], False)

# Save image
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
