#!/bin/bash

# set apt DEBIAN_FRONTEND environment variable
# mode when we need zero interation while installing or upgrading packages in the system via apt
export DEBIAN_FRONTEND=noninteractive

#  update the package lists for upgrades for packages that need upgrading, as well as new packages that have just come to the repositories
apt update

# install indent utility
apt --assume-yes install indent

# install  essential packages
apt --assume-yes install build-essential

# install  c compiler, g++ compiler and make utility
apt --assume-yes install gcc g++ make

# install GPGPU-sim dependencies
apt --assume-yes install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev

# install GPGPU-sim documentation dependencies
apt --assume-yes install doxygen graphviz

# install AerialVision dependencies
apt --assume-yes install python-pmw python-ply python-numpy libpng12-dev python-matplotlib

# install CUDA SDK dependencies
apt --assume-yes install libxi-dev libxmu-dev freeglut3-dev libxml2

# install control version system
apt --assume-yes install git

# install wget utility
apt --assume-yes install wget

# access the home directory and print the current working directory
cd $HOME
echo $HOME

# create a directory cuda
mkdir cuda
cd cuda

# dowload driver for NVIDIA
wget -P /tmp wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run

# install driver for NVIDIA
sh /tmp/cuda_10.1.243_418.87.00_linux.run --silent --toolkit --toolkitpath=$PWD
rm /tmp/cuda*

# ensure CUDA_INSTALL_PATH environment variable is set correctly and add it to the PATH environment variable
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

cd ..

# clone repository GPGPU-sim
git clone https://github.com/srirajpaul/gpgpu-sim_distribution.git

# access the GPGPU-sim directory
cd gpgpu-sim_distribution

# change the branch
git checkout docker_0.1_4.0.1

#source the setup_environment script, this will set the environment variables needed to compile and run GPGPU-Sim
source setup_environment

#build GPGPU-Sim
make
rm -rf build/
cd ..

echo "" >> .bashrc
echo "source .bashrc.ext1" >> .bashrc
echo "" >> .bashrc
echo "export PTX_SIM_MODE_FUNC=1" >> .bashrc.ext1
echo "export PTX_SIM_MODE_DETAIL=0" >> .bashrc.ext1
echo "export CUDA_INSTALL_PATH=\$HOME/cuda" >> .bashrc.ext1
echo "export PATH=\$CUDA_INSTALL_PATH/bin:\$PATH" >> .bashrc.ext1
echo "source \$HOME/gpgpu-sim_distribution/setup_environment" >> .bashrc.ext1
echo "export GPGPUSIM_ARCH_PATH=\$GPGPUSIM_ROOT/configs/tested-cfgs/SM75_RTX2060/" >> .bashrc.ext1
source .bashrc.ext1

rm /tmp/script.sh
