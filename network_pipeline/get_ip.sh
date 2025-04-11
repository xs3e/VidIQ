#!/bin/bash
net_device=$(iwconfig 2>/dev/null | awk '/^[a-zA-Z0-9]+/{print $1}')
# echo "Obtain a local wireless card ${net_device}"
while :; do
    inet=$(ifconfig | grep -A 1 "$net_device")
    OLD_IFS="$IFS"
    IFS=" "
    inets=($inet)
    IFS="$OLD_IFS"
    let "i=0"
    for var in ${inets[@]}; do
        if [ $var == "inet" ]; then
            break
        fi
        let "i=i+1"
    done
    let "i=i+1"
    local_IP=${inets[$i]}
    local_IP=$(echo "$local_IP" | sed 's/ //g')
    if [ "$local_IP" != "" ]; then
        break
    fi
done
echo $local_IP
