#!/bin/bash
input=${HOME}/.local/lib/python3.*/site-packages/imageio_ffmpeg
error=1
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ]; then
        error=0
        sudo apt install ffmpeg -y
        pip3 install imageio imageio-ffmpeg
        input=$(echo ${input})
        ffmpeg_name=$(find ${input}/binaries -type f -name 'ffmpeg-*')
        backup_name=${input}/binaries/backup
        if [ ! -f "${backup_name}" ]; then
            mv ${ffmpeg_name} ${backup_name}
            ln -s ffmpeg ${ffmpeg_name}
        fi
    elif [ $1 == "0" ]; then
        error=0
        pip3 uninstall imageio imageio-ffmpeg
        rm -rf ${input}
    fi
fi
if [ $error == "1" ]; then
    echo "./prerequisite.sh 1	Install imageio-ffmpeg"
    echo "./prerequisite.sh 0	Uninstall imageio-ffmpeg"
fi
