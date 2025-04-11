#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input/..
# loss_index = 4 && is_load=True
python3 autoencoder/train.py
python3 autoencoder/train3d.py
