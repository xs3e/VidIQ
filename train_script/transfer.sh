#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input/..
# loss_index = 4 && is_load=True && transfer="4.2.16.6"
sed -i "s#^NETWORK_DEPTH_3D = 4#NETWORK_DEPTH_3D = 3#g" autoencoder/network3d.py
python3 autoencoder/train3d.py
sed -i "s#^NETWORK_DEPTH_3D = 3#NETWORK_DEPTH_3D = 2#g" autoencoder/network3d.py
python3 autoencoder/train3d.py
sed -i "s#^NETWORK_DEPTH_3D = 2#NETWORK_DEPTH_3D = 5#g" autoencoder/network3d.py
python3 autoencoder/train3d.py
sed -i "s#^NETWORK_DEPTH_3D = 5#NETWORK_DEPTH_3D = 4#g" autoencoder/network3d.py
# loss_index = 2 && is_load=True && transfer="2"
sed -i "s#^SCALE_RATIO = 2#SCALE_RATIO = 3#g" super_resolution/models.py
python3 super_resolution/train3d.py
sed -i "s#^SCALE_RATIO = 3#SCALE_RATIO = 4#g" super_resolution/models.py
python3 super_resolution/train3d.py
sed -i "s#^SCALE_RATIO = 4#SCALE_RATIO = 2#g" super_resolution/models.py
