# VidIQ: Inference-Aware Neural Codecs for Quality-Enhanced, Real-Time Video Analytics

This repository hosts the prototype implementation of our paper *VidIQ: Inference-Aware Neural Codecs for Quality-Enhanced, Real-Time Video Analytics*.

## Prerequisites

Python 3.8.10  
CUDA 11.7.1  
PyTorch 2.0.1  
OpenCV 4.8.0  
ffmpeg 7:4.2.7-0ubuntu0.1  
torchmetrics 1.1.0  
pytorch-msssim 1.0.0  
pycocotools 2.0.7  
screen 4.8.0-1ubuntu0.1  
imageio-ffmpeg 0.4.9  
openssh-client 1:8.2p1-4ubuntu0.11  
openssh-server 1:8.2p1-4ubuntu0.11  
sshpass 1.06-1  
jq 1.6-1ubuntu0.20.04.1  

## Install Instructions

To deploy our code, first execute  
``cd VidIQ;./configure.sh 1``  
to configure environment variables.

Before running *VidIQ*, you need to download dataset *youtube-objects* and *DIV2K* to the directory *DOWNLOAD_DIR* identified by ``autoencoder/util.py``,
Similarly, download DNN model weights corresponding to analytics task to this directory, and execute ``./dnn_model/weights.sh`` to install them.

Now, we can execute the following commands to train *Layered Encoder* and *SR-Decoder*  
``cd train_script;./base.sh;./loss.sh;./joint.sh``  
(all training scripts turn on DDP mode by default to squeeze multiple GPUs, and visit function *resize_video_for_limited_memory_and_downsample* in ``autoencoder/dataset.py`` to accommodate a memory constrained GPU)  
The trained codec weights are automatically saved in the directory *DOWNLOAD_DIR*.

In addition, we can use ``train_script/transfer.py`` to transfer the trained network weights to other new structures (to support different compression and SR factors), and ``super_resolution/models.py`` shows two specific integration cases of *Layered Decoder* and SR model.

Note that *Monolithic Controller* uses the accuracy fitting results for three default DNNs (i.e., YOLOv5m, MaskRCNN, and KeypointRCNN), visit *compression_profiling* function and *enhancement_profiling* function in ``network_pipeline/monolithic_controller.py`` to perform a new fit on other DNNs.

After the training is completed, you can run the following command to launch *VidIQ*  
``python3 preheat.py``

To enable real-time network transmission with scenario adaptation, execute ``network_pipeline/client.py`` and ``network_pipeline/server.py`` (*Monolithic Controller* is automatically launched to control the behavior of *Layered Encoder* and *SR-Decoder*).
