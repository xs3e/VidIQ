#!/bin/bash
if [ "$LANG" == "zh_CN.UTF-8" ]; then
    media="/media/${USER}/数据硬盘/下载"
    if [[ -d "${media}" && -n "$(ls -A ${media})" ]]; then
        DOWNLOAD_DIR="${media}"
    else
        DOWNLOAD_DIR="/media/${USER}/PortableSSD/下载"
    fi
else
    DOWNLOAD_DIR="/data/${USER}/crucio_downloads"
fi
if [[ ! -d "${DOWNLOAD_DIR}/youtube-objects" ]]; then
    mkdir ${DOWNLOAD_DIR}/youtube-objects
    cd ${DOWNLOAD_DIR}/youtube-objects
    DATASET_URL="https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories"
    curl -L -O ${DATASET_URL}/aeroplane.tar.gz
    curl -L -O ${DATASET_URL}/bird.tar.gz
    curl -L -O ${DATASET_URL}/boat.tar.gz
    curl -L -O ${DATASET_URL}/car.tar.gz
    curl -L -O ${DATASET_URL}/cat.tar.gz
    curl -L -O ${DATASET_URL}/cow.tar.gz
    curl -L -O ${DATASET_URL}/dog.tar.gz
    curl -L -O ${DATASET_URL}/horse.tar.gz
    curl -L -O ${DATASET_URL}/motorbike.tar.gz
    curl -L -O ${DATASET_URL}/train.tar.gz
fi
echo ${DOWNLOAD_DIR}
