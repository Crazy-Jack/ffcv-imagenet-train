

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
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/workspace/env/Install-OpenCV/source/lib/pkgconfig
# LD_LIBRARY_PATH
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

