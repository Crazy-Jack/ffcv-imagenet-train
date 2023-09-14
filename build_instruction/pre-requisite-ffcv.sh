#!/bin/sh 

# try to find out which folder the follow exist
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR=$1
cd $WORKDIR
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda init bash
conda activate 

# for ffcv environment
conda create -n ffcv python=3.9 -y
conda activate ffcv
# for torch
pip3 install torch torchvision torchaudio

CUDAVERSION=$2
pip3 install cupy-cuda$2 # choose from 11x or 12x
conda install -c conda-forge libstdcxx-ng -y
pip3 install numba

MY_OPENCV_INSTALL_FOLDER_DIR=$WORKDIR/Install-OpenCV
mkdir $MY_OPENCV_INSTALL_FOLDER_DIR
cd $MY_OPENCV_INSTALL_FOLDER_DIR
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
mv opencv-4.x opencv
cd opencv 
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$MY_OPENCV_INSTALL_FOLDER_DIR/source/ -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D PYTHON_EXECUTABLE=$(which python2) -D BUILD_opencv_python2=OFF -D PYTHON3_EXECUTABLE=$(which python3) -D OPENCV_GENERATE_PKGCONFIG=ON -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. 
make -j8
make install 
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$MY_OPENCV_INSTALL_FOLDER_DIR/source/lib/pkgconfig
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$MY_OPENCV_INSTALL_FOLDER_DIR/source/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MY_OPENCV_INSTALL_FOLDER_DIR/source/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MY_OPENCV_INSTALL_FOLDER_DIR/source/lib64

MY_LIBTURB_FOLDER_DIR=$WORKDIR/Install-libjpeg-turbo
mkdir MY_LIBTURB_FOLDER_DIR
cd $MY_LIBTURB_FOLDER_DIR
wget https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/heads/main.zip
unzip main.zip 
mv libjpeg-turbo-main libjpeg-turbo
cd libjpeg-turbo
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$MY_LIBTURB_FOLDER_DIR/install/ ..
make -j8
make install

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$MY_LIBTURB_FOLDER_DIR/install/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MY_LIBTURB_FOLDER_DIR/install/lib/

# import ffcv
pip install ffcv

pip install pyyaml

pip install /workspace/ffcv-imagenet-train/build_instruction/torchmetrics-0.6.0.tar.gz