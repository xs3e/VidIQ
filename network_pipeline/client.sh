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
    echo -e "\e[31mNote: Run client.sh on client!\e[0m"
    exit
fi

input_path=$(../video_data/download.sh)
echo ${input_path}
if [ ! -d "${input_path}" ]; then
    echo "Directory ${input_path} does not exist"
    exit
fi

sudo apt install sshpass -y
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d /data/$USERNAME ]"; then
    echo "/data/$USERNAME does not exist in host ${HOST}"
    exit
fi
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d $DOWNLOAD_DIRS ]"; then
    echo "$USERNAME@$HOST:$DOWNLOAD_DIRS"
    sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "mkdir $DOWNLOAD_DIRS"
fi
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d $DOWNLOAD_DIRS/val2017 ]"; then
    echo "$USERNAME@$HOST:$DOWNLOAD_DIRS/val2017"
    sshpass -p $PASSWORD scp -v -r $input_path/val2017 $USERNAME@$HOST:$DOWNLOAD_DIRS/val2017
fi
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d $DOWNLOAD_DIRS/youtube-objects ]"; then
    echo "$USERNAME@$HOST:$DOWNLOAD_DIRS/youtube-objects"
    sshpass -p $PASSWORD scp -v -r $input_path/youtube-objects $USERNAME@$HOST:$DOWNLOAD_DIRS/youtube-objects
fi
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d /home/$USERNAME/.cache/torch ]"; then
    echo "$USERNAME@$HOST:/home/$USERNAME/.cache/torch/*"
    sshpass -p $PASSWORD scp -v -r ${HOME}/.cache/torch $USERNAME@$HOST:/home/$USERNAME/.cache/torch
fi
