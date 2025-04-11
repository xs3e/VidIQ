#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
HOST=$(cat config.json | jq -r '.host')
USERNAME=$(cat config.json | jq -r '.username')
VIDEO_DATASET="youtube-objects"
DOWNLOAD_DIRS="/data/$USERNAME/crucio_downloads/$VIDEO_DATASET"
PASSWORD=$(cat config.json | jq -r '.password')

local_IP=$(./get_ip.sh)
if [ "$local_IP" == "$HOST" ]; then
    echo -e "\e[31mNote: Run syn_dataset.sh on client!\e[0m"
    exit
fi

input_path=$(../video_data/download.sh)"/$VIDEO_DATASET"
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
                    sshpass -p $PASSWORD scp -r $input_path/$file $USERNAME@$HOST:$DOWNLOAD_DIRS/$file
                    echo "[$count] Copied $input_path/$file to $USERNAME@$HOST:$DOWNLOAD_DIRS/$file"
                fi
            done
        elif [ "$1" == "0" ]; then
            for file in $server_files; do
                if [[ ! "${client_files[@]}" =~ "$file" ]]; then
                    count=$((count + 1))
                    sshpass -p $PASSWORD scp -r $USERNAME@$HOST:$DOWNLOAD_DIRS/$file $input_path/$file
                    echo "[$count] Copied $USERNAME@$HOST:$DOWNLOAD_DIRS/$file to $input_path/$file"
                fi
            done
        fi
        server_count=$(sshpass -p $PASSWORD ssh "$USERNAME@$HOST" "ls -l $DOWNLOAD_DIRS | grep "^d" | wc -l")
        client_count=$(ls -l $input_path | grep "^d" | wc -l)
        if [ "$1" == "1" ]; then
            echo "Find $client_count videos in $input_path"
            echo "Add $count videos to $USERNAME@$HOST:$DOWNLOAD_DIRS to reach $server_count"
        elif [ "$1" == "0" ]; then
            echo "Find $server_count videos in $USERNAME@$HOST:$DOWNLOAD_DIRS"
            echo "Add $count videos to $input_path to reach $client_count"
        fi
    fi
fi
if [ $error == "1" ]; then
    echo "./syn_video.sh 1	Transfer dataset from local to server"
    echo "./syn_video.sh 0	Transfer dataset from server to local"
fi
