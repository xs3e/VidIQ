#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
HOST=$(cat config.json | jq -r '.host')
USERNAME=$(cat config.json | jq -r '.username')
IMAGE_DATASET="val2017"
DOWNLOAD_DIRS="/data/$USERNAME/crucio_downloads/$IMAGE_DATASET"
PASSWORD=$(cat config.json | jq -r '.password')

local_IP=$(./get_ip.sh)
if [ "$local_IP" == "$HOST" ]; then
    echo -e "\e[31mNote: Run syn_dataset.sh on client!\e[0m"
    exit
fi

input_path=$(../video_data/download.sh)"/$IMAGE_DATASET"
echo ${input_path}
if [ ! -d "${input_path}" ]; then
    echo "Directory ${input_path} does not exist"
    exit
fi

error=1
sudo apt install sshpass -y
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ] || [ "$1" == "0" ]; then
        error=0
        server_files=$(sshpass -p $PASSWORD ssh "$USERNAME@$HOST" "ls $DOWNLOAD_DIRS")
        client_files=$(ls $input_path)
        count=0
        if [ "$1" == "1" ]; then
            for file in $client_files; do
                if [[ ! "${server_files[@]}" =~ "$file" ]]; then
                    count=$((count + 1))
                    sshpass -p $PASSWORD scp $input_path/$file $USERNAME@$HOST:$DOWNLOAD_DIRS/$file
                    echo "[$count] Copied $input_path/$file to $USERNAME@$HOST:$DOWNLOAD_DIRS/$file"
                fi
            done
        elif [ "$1" == "0" ]; then
            for file in $server_files; do
                if [[ ! "${client_files[@]}" =~ "$file" ]]; then
                    count=$((count + 1))
                    sshpass -p $PASSWORD scp $USERNAME@$HOST:$DOWNLOAD_DIRS/$file $input_path/$file
                    echo "[$count] Copied $USERNAME@$HOST:$DOWNLOAD_DIRS/$file to $input_path/$file"
                fi
            done
        fi
        server_count=$(sshpass -p $PASSWORD ssh "$USERNAME@$HOST" "ls -l $DOWNLOAD_DIRS | grep "^-" | wc -l")
        client_count=$(ls -l $input_path | grep "^-" | wc -l)
        if [ "$1" == "1" ]; then
            echo "Find $client_count images in $input_path"
            echo "Add $count images to $USERNAME@$HOST:$DOWNLOAD_DIRS to reach $server_count"
        elif [ "$1" == "0" ]; then
            echo "Find $server_count images in $USERNAME@$HOST:$DOWNLOAD_DIRS"
            echo "Add $count images to $input_path to reach $client_count"
        fi
    fi
fi
if [ $error == "1" ]; then
    echo "./syn_image.sh 1	Transfer dataset from local to server"
    echo "./syn_image.sh 0	Transfer dataset from server to local"
fi
