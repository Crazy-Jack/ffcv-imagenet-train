# how to deploy to the server

RunPod Pytorch 2.0.1
ID: 23x54s4ra3apuc
1 x NVIDIA L40
32 vCPU 250 GB RAM
runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel
On-Demand - Secure Cloud
Running
100 GB Disk
500 GB Pod Volume
Volume Path: /workspace



-----------------------
-  Second Time
-----------------------
```
apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        ffmpeg \
        tmux \
        cmake \
        g++ wget unzip \
        pkg-config \
        vim
```

this gives you tmux 

go to a tmux window
ENV variable
```
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/workspace/env/Install-OpenCV/source/lib/pkgconfig
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/workspace/env/Install-libjpeg-turbo/install/lib/pkgconfig
```
# configure LD_library
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/env/Install-OpenCV/source/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/env/Install-libjpeg-turbo/install/lib/
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
# configure MODELVSHUMANDIR for on-fly shape score evaluation
```
export MODELVSHUMANDIR=/workspace/ffcv-imagenet-train/model-vs-human
```
# OR on private server ==========
```
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/ylz1122/Install-OpenCV-2/source/lib/pkgconfig
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/ylz1122/.local/libturbojpeg-2/lib/pkgconfig

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ylz1122/Install-OpenCV-2/source/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ylz1122/.local/libturbojpeg-2/lib
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

export MODELVSHUMANDIR=/home/ylz1122/ffcv-imagenet-train/model-vs-human
```

# conda
```
source /workspace/env/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate ffcv
```

# ssh
```
ln -s /workspace/.ssh/* ~/.ssh
chmod 0600 ~/.ssh/id_rsa
```


-----------------------
- Install from scratch 
-----------------------
0. System install 
```
apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        ffmpeg \
        tmux \
        cmake \
        g++ wget unzip \
        pkg-config \
        vim

# ENV variable
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/workspace/env/Install-OpenCV/source/lib/pkgconfig
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/workspace/env/Install-libjpeg-turbo/install/lib/pkgconfig

# configure LD_library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/env/Install-OpenCV/source/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/env/Install-libjpeg-turbo/install/lib/
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

source /workspace/env/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate ffcv

```

# download data 
0. download the dataset 

Use backblaze for cheap version of S3
```
mkdir -p /workspace/data && cd /workspace/data
wget https://github.com/Backblaze/B2_Command_Line_Tool/releases/latest/download/b2-linux
mv b2-linux b2
chmod +x b2 

./b2 authorize_account
Backblaze application key ID: 
Backblaze application key: 

./b2 download_file_by_id 4_z9bfa244ea904c331819a0518_f2013f5c2d42f1c1e_d20230805_m133806_c005_v0501007_t0054_u01691242686010 train_500_0.50_90.ffcv

./b2 download_file_by_id 4_z9bfa244ea904c331819a0518_f209b1e9245eb6a42_d20230805_m215628_c005_v0501000_t0025_u01691272588680 val_500_0.50_90.ffcv

```
--------------------


1. Install conda 
```
mkdir -p /workspace/env/ && cd /workspace/env/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

>>> /workspace/env/miniconda3

# the following activates conda everything you exit and re-enter 
```
source /workspace/env/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate 
conda create -n ffcv python=3.9 -y
conda activate ffcv

```


since the /workspace/env is persistence disk, the conda file will be saved 



6. Install torch 
```
pip3 install torch torchvision torchaudio

```



7. Install CUPY 

To install CUPY, you need to specify based on the CUDA version.

`$ nvcc --version`
we requested 11.8 so this will output
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```


- v11.2 ~ 11.8 (x86_64 / aarch64)
`pip3 install cupy-cuda11x`

- v12.x (x86_64 / aarch64)
`pip3 install cupy-cuda12x`

therefore we will use `pip3 install cupy-cuda11x`

* don't forget to  setup the env for cuda

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH



9. libstd++
`conda install -c conda-forge libstdcxx-ng -y
`

10. install numba
`pip install numba`

11. Compile opencv 



- [SYSTEM Reinstall] install cmake etc
```
apt update && apt install -y cmake g++ wget unzip

- Install opencv from source 
```
```
# Integreate code from torchinstall section

pip3 install torch torchvision torchaudio

pip3 install cupy-cuda11x

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

conda install -c conda-forge libstdcxx-ng -y

pip install numba
# start opencv
apt update && apt install -y cmake g++ wget unzip

# Download and unpack sources
# start from 4:25 - 4:39 finished - install opencv from source takes 14 mins.
cd /workspace/env/
mkdir Install-OpenCV
cd Install-OpenCV
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
mv opencv-4.x opencv

cd opencv 
# Create build directory
mkdir -p build && cd build

# cmake 
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/workspace/env/Install-OpenCV/source/ -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D PYTHON_EXECUTABLE=$(which python2) -D BUILD_opencv_python2=OFF -D PYTHON3_EXECUTABLE=$(which python3) -D OPENCV_GENERATE_PKGCONFIG=ON -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \

make -j8
make install 

# Configure PKG
# export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/workspace/env/Install-OpenCV/opencv/build/unix-install/  

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/workspace/env/Install-OpenCV/source/lib/pkgconfig
# LD_LIBRARY_PATH
# configure LD_library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/env/Install-OpenCV/source/lib


# Compile libturbojpeg

mkdir -p /workspace/env/Install-libjpeg-turbo/ && cd /workspace/env/Install-libjpeg-turbo/
wget https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/heads/main.zip
unzip main.zip 
mv libjpeg-turbo-main libjpeg-turbo
cd libjpeg-turbo

mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/workspace/env/Install-libjpeg-turbo/install/ ..
make -j8
make install

# PKG_CONFIG_PATH
# export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/workspace/env/Install-libjpeg-turbo/libjpeg-turbo/build/pkgscripts
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/workspace/env/Install-libjpeg-turbo/install/lib/pkgconfig
# configure LD_library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/env/Install-libjpeg-turbo/install/lib/

# install ffcv
pip install ffcv


# Download source code for training 


cd /workspace/
git clone https://github.com/Crazy-Jack/ffcv-imagenet-train.git


# 14. install torchmetrics

pip install /workspace/ffcv-imagenet-train/build_instruction/torchmetrics-0.6.0.tar.gz


# install local modified modules


cd /workspace/ffcv-imagenet-train/vit-pytorch-customized
pip install -e .


# 16. install yaml


pip install pyyaml

17. Install ModelVsHuman and ShapeBiasEval

cd /workspace/ffcv-imagenet-train/model-vs-human
pip install -e .

cd /workspace/ffcv-imagenet-train/shape_bias
pip install -e .

export MODELVSHUMANDIR=/workspace/ffcv-imagenet-train/model-vs-human


18. matplotlib

```
pip install matplotlib
```
17. Debug: 

[W socket.cpp:601] [c10d] The client socket has failed to connect to [localhost]:12321 (errno: 99 - Cannot assign requested address).

- solution: need to expose internal TCP port, for example 12321, 12320, etc and use these port when doing distributed training 

<!-- 18. RandAug

```
pip install git+https://github.com/ildoonet/pytorch-randaugment
```
 -->





Appendix:

For aws:
```
conda create -n aws python=3.9 -y
conda activate aws
pip install awscli

aws configure

Input the following ID and key to set the aws configure.

Access key ID: AKIA3WS7UL243YIUYHOJ
Secret access key: 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

Other input can be empty.

mkdir -p ffcv-image 
# start from 1:24 
aws s3 cp s3://imagenetcache/ImagenetFFCV/train_500_0.50_90.ffcv ffcv-image/
aws s3 cp s3://imagenetcache/ImagenetFFCV/val_500_0.50_90.ffcv ffcv-image/
```
