#!/bin/bash

sudo apt update
sudo apt install python3-pip openjdk-18-jdk
pip3 install pydot networkx tqdm numpy

cd prism
./install.sh


