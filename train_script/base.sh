#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input/..
# loss_index = 2 && is_load=False
python3 autoencoder/train.py
python3 super_resolution/train.py
python3 autoencoder/train3d.py
python3 super_resolution/train3d.py
