#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input/..
# (1) loss_index = 2 && is_load=False
python3 joint_training/train3d.py
sed -i "s#CODEC_MODE = \"encoder\"#CODEC_MODE = \"decoder\"#g" joint_training/train3d.py
python3 joint_training/train3d.py
sed -i "s#CODEC_MODE = \"decoder\"#CODEC_MODE = \"encoder\"#g" joint_training/train3d.py
# (2) loss_index = 4 && is_load=True
sed -i "s#^loss_index = 2#loss_index = 4#g" joint_training/train3d.py
sed -i "s#get_networks3d(is_load=False#get_networks3d(is_load=True#g" joint_training/train3d.py
sed -i "s#get_models(is_load=False#get_models(is_load=True#g" joint_training/train3d.py
python3 joint_training/train3d.py
sed -i "s#CODEC_MODE = \"encoder\"#CODEC_MODE = \"decoder\"#g" joint_training/train3d.py
python3 joint_training/train3d.py
sed -i "s#CODEC_MODE = \"decoder\"#CODEC_MODE = \"encoder\"#g" joint_training/train3d.py
sed -i "s#^loss_index = 4#loss_index = 2#g" joint_training/train3d.py
sed -i "s#get_networks3d(is_load=True#get_networks3d(is_load=False#g" joint_training/train3d.py
sed -i "s#get_models(is_load=True#get_models(is_load=False#g" joint_training/train3d.py
