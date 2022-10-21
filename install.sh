#!/bin/bash

# Install required software
sudo apt update
sudo apt install git python3 python3-pip openjdk-17-jdk xdot
pip3 install pydot networkx tqdm numpy dtcontrol regex

# Download and install PRISM model checker
wget https://www.prismmodelchecker.org/dl/prism-4.7-linux64.tar.gz#download-box
mkdir prism
tar -zxf prism-4.7-linux64.tar.gz --strip-components=1 -C prism
cd prism
./install.sh
cd -
