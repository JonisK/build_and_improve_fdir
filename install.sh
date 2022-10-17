#!/bin/bash


sudo apt update
sudo apt install git python3.8 python3-pip openjdk-17-jdk
pip3 install xdot pydot networkx tqdm numpy

# Download and install PRISM model checker
git clone https://github.com/prismmodelchecker/prism.git
cd prism/prism
make
./install.sh

