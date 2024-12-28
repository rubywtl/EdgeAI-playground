#!/bin/bash

script hw1.script
mkdir -p HW1
cd HW1
pwd
echo "Linux is great"
echo "Linux is great" > now
date >> now
uname -r >> now
wc -l < now
cat now
cp now pastnow
cd ~
find . -name pastnow
cd ~/cse374/HW1
ls -l
chmod 754 now

echo "alias ll='ls -l'" >> ~/.bashrc
source ~/.bashrc
ll
emacs &

# Step 21: Suspend emacs using 'Ctrl + Z'
# This step will require manual input to suspend with Ctrl + Z
# You will use 'fg' after this

# Step 22: List all processes belonging to the user
ps -u $USER

# Step 23: Filter the process list using grep for 'emacs'
ps -u $USER | grep emacs

# Step 24: Kill the emacs process using signal -9
kill -9 $(ps -u $USER | grep emacs | awk '{print $1}')

# Step 25: Use echo and the environment variable USER to print "user is great"
echo "$USER is great"

# Step 26: End the script recording
exit
