#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
HOST=$(cat config.json | jq -r '.host')
USERNAME=$(cat config.json | jq -r '.username')
DOWNLOAD_DIRS="/data/$USERNAME/crucio_downloads"
PASSWORD=$(cat config.json | jq -r '.password')

local_IP=$(./get_ip.sh)
if [ "$local_IP" == "$HOST" ]; then
    echo -e "\e[31mNote: Run weights.sh on client!\e[0m"
    exit
fi

input_path=$(../video_data/download.sh)
echo ${input_path}
if [ ! -d "${input_path}" ]; then
    echo "Directory ${input_path} does not exist"
    exit
fi

NETWORK_DEPTH=$(grep -m 1 '^NETWORK_DEPTH' ../autoencoder/network2d.py | grep -oP '^NETWORK_DEPTH\s*=\s*\K[^,]*')
FEATURE_CHANNEL=$(grep -m 1 '^FEATURE_CHANNEL' ../autoencoder/network2d.py | grep -oP '^FEATURE_CHANNEL\s*=\s*\K[^,]*')
BITS_CHANNEL=$(grep -m 1 '^BITS_CHANNEL' ../autoencoder/network2d.py | grep -oP '^BITS_CHANNEL\s*=\s*\K[^,]*')
WEIGHTS_2D="weights_network_"$NETWORK_DEPTH.$FEATURE_CHANNEL.$BITS_CHANNEL

NETWORK_DEPTH_3D=$(grep -m 1 '^NETWORK_DEPTH_3D' ../autoencoder/network3d.py | grep -oP '^NETWORK_DEPTH_3D\s*=\s*\K[^,]*')
FEATURE_CHANNEL_3D=$(grep -m 1 '^FEATURE_CHANNEL_3D' ../autoencoder/network3d.py | grep -oP '^FEATURE_CHANNEL_3D\s*=\s*\K[^,]*')
BITS_CHANNEL_3D=$(grep -m 1 '^BITS_CHANNEL_3D' ../autoencoder/network3d.py | grep -oP '^BITS_CHANNEL_3D\s*=\s*\K[^,]*')
CNN_FRAME_NUM=$(grep -m 1 '^CNN_FRAME_NUM' ../autoencoder/dataset.py | grep -oP '^CNN_FRAME_NUM\s*=\s*\K[^,]*')
WEIGHTS_3D="weights_network3d_"$NETWORK_DEPTH_3D.$FEATURE_CHANNEL_3D.$BITS_CHANNEL_3D.$CNN_FRAME_NUM

MODEL_NAME=$(grep -m 1 '^MODEL_NAME' ../super_resolution/models.py | grep -oP '^MODEL_NAME\s*=\s*"\K[^"]*')
SCALE_RATIO=$(grep -m 1 '^SCALE_RATIO' ../super_resolution/models.py | grep -oP '^SCALE_RATIO\s*=\s*\K[^,]*')
SR_MODEL=$MODEL_NAME"_x"$SCALE_RATIO

error=1
sudo apt install sshpass -y
LOCAL_RGB_WEIGHTS=$input_path/"rgb_weights"
LOCAL_YUV_WEIGHTS=$input_path/"yuv_weights"
SERVER_RGB_WEIGHTS=$DOWNLOAD_DIRS/"rgb_weights"
SERVER_YUV_WEIGHTS=$DOWNLOAD_DIRS/"yuv_weights"
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ]; then
        error=0
        if [ -d $LOCAL_RGB_WEIGHTS ]; then
            sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "rm -rf $SERVER_RGB_WEIGHTS"
            sshpass -p $PASSWORD scp -v -r $LOCAL_RGB_WEIGHTS $USERNAME@$HOST:$SERVER_RGB_WEIGHTS
        fi
        if [ -d $LOCAL_YUV_WEIGHTS ]; then
            sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "rm -rf $SERVER_YUV_WEIGHTS"
            sshpass -p $PASSWORD scp -v -r $LOCAL_YUV_WEIGHTS $USERNAME@$HOST:$SERVER_YUV_WEIGHTS
        fi
    elif [ $1 == "0" ]; then
        error=0
        if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ -d $SERVER_RGB_WEIGHTS ]"; then
            rm -rf $LOCAL_RGB_WEIGHTS
            sshpass -p $PASSWORD scp -v -r $USERNAME@$HOST:$SERVER_RGB_WEIGHTS $LOCAL_RGB_WEIGHTS
        fi
        if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ -d $SERVER_YUV_WEIGHTS ]"; then
            rm -rf $LOCAL_YUV_WEIGHTS
            sshpass -p $PASSWORD scp -v -r $USERNAME@$HOST:$SERVER_YUV_WEIGHTS $LOCAL_YUV_WEIGHTS
        fi
    fi
fi
if [ $error == "1" ]; then
    echo $WEIGHTS_2D
    echo $WEIGHTS_3D
    echo $SR_MODEL
    echo "./weights.sh 1	Transfer trained weights from local to server"
    echo "./weights.sh 0	Transfer trained weights from server to local"
fi
