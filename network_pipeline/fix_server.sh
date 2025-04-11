#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
error=1

clientpass=$(cat config.json | jq -r '.clientpass')
if [ $clientpass == null ]; then
	echo "Key \"clientpass\" not found in config.json"
	exit
fi
HOST=$(cat config.json | jq -r '.host')
if [ $HOST == null ]; then
	echo "Key \"host\" not found in config.json"
	exit
fi
local_IP=$(./get_ip.sh)
if [ "$local_IP" == "$HOST" ]; then
	echo -e "\e[31mNote: Run fix_server.sh on client!\e[0m"
	exit
fi
shell=$(cat config.json | jq -r '.shell')
if [ $shell == null ]; then
	echo "Key \"shell\" not found in config.json"
	exit
fi
USERNAME=$(cat config.json | jq -r '.username')
if [ $USERNAME == null ]; then
	echo "Key \"username\" not found in config.json"
	exit
fi
PASSWORD=$(cat config.json | jq -r '.password')
if [ $PASSWORD == null ]; then
	echo "Key \"password\" not found in config.json"
	exit
fi

if [ $shell == "zsh" ]; then
	rc="/home/$USERNAME/.zshrc"
else
	rc="/home/$USERNAME/.bashrc"
fi
sshconfig="/home/$USERNAME/.ssh"
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d $sshconfig ]"; then
	sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "mkdir $sshconfig"
fi
known_hosts="$sshconfig/known_hosts"
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -f $known_hosts ]"; then
	sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "touch $known_hosts"
fi

client_IP=$(./get_ip.sh)
echo "Obtain local IPv4 address "$client_IP
client_user=${USER}
bind_port=1081

if [ -n "$1" ] && [ ! -n "$2" ]; then
	tunnel=0
	pid=$(sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "pidof ssh")
	if [ "$pid" ]; then
		u=$(sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "ps -p $pid -o user=")
		if [ $u = $USERNAME ]; then
			tunnel=1
		fi
	fi
	known=0
	find=$(sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \
		"grep \"$client_IP\" $known_hosts")
	if [ "$find" ]; then
		known=1
	fi
	sshd=$(ps -e | grep sshd)
	if [ $1 == "1" ]; then
		error=0
		sudo apt-get install openssh-client openssh-server -y
		if [ -z "$sshd" ]; then
			sudo ufw allow ssh
			sudo service ssh start
			echo "Start local SSH server"
		fi
		if [ "$known" == "0" ]; then
			sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \
				"ssh-keyscan -t rsa $client_IP >>$known_hosts"
			echo "Local HOST has been added to known_hosts in ${USERNAME}@${HOST}"
		fi
		if [ "$tunnel" == "0" ]; then
			sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \
				"sshpass -p $clientpass ssh -D $bind_port -q -C -N -f $client_user@$client_IP" &
			sleep 3
			pid=$(pidof sshpass)
			if [ "$pid" ]; then
				kill $pid
			fi
			echo "SOCKS5 SSH tunnel of ${USERNAME}@${HOST} has been enabled"
		fi
		find=$(sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \
			"grep \"socks5h://localhost:$bind_port\" $rc")
		if [ -z "$find" ]; then
			sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \
				"echo "export http_proxy=socks5h://localhost:$bind_port" >>$rc; \
			echo "export https_proxy=socks5h://localhost:$bind_port" >>$rc; \
			echo "export ftp_proxy=socks5h://localhost:$bind_port" >>$rc;source $rc"
			echo "Proxy has been successfully set for ${USERNAME}@${HOST}"
		fi
		echo "Network connection for ${USERNAME}@${HOST} has been fixed"
		echo -e "\e[34mRun following command in new SSH terminal to test connectivity\e[0m"
		echo "sudo apt-get install curl -y;curl www.baidu.com"

	elif [ $1 == "0" ]; then
		error=0
		if [ "$tunnel" == "1" ]; then
			sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \
				"kill $pid"
			echo "SOCKS5 SSH tunnel of ${USERNAME}@${HOST} has been disabled"
		fi
		if [ "$known" == "1" ]; then
			sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \
				"ssh-keygen -R $client_IP"
			echo "Local HOST has been deleted from known_hosts in ${USERNAME}@${HOST}"
		fi
		sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \
			"sed -i "/localhost:$bind_port/d" $rc;source $rc"
		echo "Proxy Settings for ${USERNAME}@${HOST} have been deleted"
		if [ "$sshd" ]; then
			sudo service ssh stop
			echo "Close local SSH server"
		fi
	fi
fi
if [ $error == "1" ]; then
	echo -e "\e[31mClient and server must be directly connected over same LAN\e[0m"
	echo "./fix_server.sh 1	Repair server's network connection"
	echo "./fix_server.sh 0	Undo all changes on server"
fi
