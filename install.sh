#!/bin/bash

# Install required software
sudo apt update
sudo apt install git python3 python3-venv python3-pip openjdk-17-jdk xdot
sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
python3 -m venv python3-venv
source python3-venv/bin/activate
pip install pycairo PyGObject
pip install xdot pydot regex networkx tqdm numpy dtcontrol

# Download and install PRISM model checker
wget https://www.prismmodelchecker.org/dl/prism-4.7-linux64.tar.gz#download-box
mkdir prism
tar -zxf prism-4.7-linux64.tar.gz --strip-components=1 -C prism
cd prism
./install.sh
cd -
