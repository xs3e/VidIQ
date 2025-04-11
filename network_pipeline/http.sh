#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
input_path=$(../video_data/download.sh)
http_port=$(cat config.json | jq -r '.http_port')
cd $input_path
python3 -m http.server $http_port
