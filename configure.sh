#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
HOST=$(cat network_pipeline/config.json | jq -r '.host')
error=1

SERVER=0
local_IP=$(./network_pipeline/get_ip.sh)
if [ "$local_IP" == "$HOST" ]; then
    SERVER=1
fi

shell=$(env | grep SHELL=)
echo $shell
if [ $shell == "SHELL=/bin/bash" ]; then
    rc=.bashrc
else
    rc=.zshrc
fi
vidiq=$(pwd)
parent=$(dirname $vidiq)
# gedit ${HOME}/.bashrc
# gedit ${HOME}/.zshrc
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ]; then
        error=0
        echo "Configure VidIQ"
        if [ $(grep -c "# $vidiq" ${HOME}/$rc) -eq 0 ]; then
            sed -i "\$a # $vidiq" ${HOME}/$rc
            if [ $(grep -c "export PYTHONPATH=\${PYTHONPATH}:$parent$" ${HOME}/$rc) -eq 0 ]; then
                sed -i "\$a export PYTHONPATH=\${PYTHONPATH}:$parent" ${HOME}/$rc
            fi
        fi
        if [ "$SERVER" == "0" ]; then
            if [ $shell == "SHELL=/bin/bash" ]; then
                gnome-terminal -- /bin/sh -c 'source ${HOME}/$rc;exit'
            else
                gnome-terminal -- /bin/zsh -c 'source ${HOME}/$rc;exit'
            fi
        else
            echo "Please run source ${HOME}/$rc manually"
        fi
        echo "echo \${PYTHONPATH}"
    elif [ $1 == "0" ]; then
        error=0
        echo "Delete configure"
        vidiq=$(echo "${vidiq##*/}")
        sed -i "/$vidiq/d" ${HOME}/$rc
        sed -i "\,export PYTHONPATH=\${PYTHONPATH}:$parent$,d" ${HOME}/$rc
        if [ "$SERVER" == "0" ]; then
            if [ $shell == "SHELL=/bin/bash" ]; then
                gnome-terminal -- /bin/sh -c 'source ${HOME}/$rc;exit'
            else
                gnome-terminal -- /bin/zsh -c 'source ${HOME}/$rc;exit'
            fi
        else
            echo "Please run source ${HOME}/$rc manually"
        fi
        echo "echo \${PYTHONPATH}"
    fi
fi
if [ $error == "1" ]; then
    echo "./configure.sh 1	Configure VidIQ"
    echo "./configure.sh 0	Delete configure"
fi
