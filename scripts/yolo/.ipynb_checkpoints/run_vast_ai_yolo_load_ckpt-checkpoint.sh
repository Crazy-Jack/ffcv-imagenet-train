pip install pytorch_pretrained_vit

pip uninstall torchmetrics -y
pip install torchmetrics

export ENV_NAME=ffcv && \
export WORK_DIR=/workspace && \
export WORK_ENV_DIR=$WORK_DIR/env && \
export ENV_NAME=$WORK_ENV_DIR/miniconda/envs/ffcv && \
source activate $ENV_NAME && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib && \
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$WORK_ENV_DIR/Install-OpenCV/source/lib/pkgconfig && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORK_ENV_DIR/Install-OpenCV/source/lib && \
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$WORK_ENV_DIR/Install-libjpeg-turbo/install/lib/pkgconfig && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORK_ENV_DIR/Install-libjpeg-turbo/install/lib/ && \

# 8 GPU training (use only 1 for ResNet-18 training)
export CUDA_VISIBLE_DEVICES=0,1;




export CURRENT_DIR=$PWD

# download models 
EXP_NAME=$1
FILENAME=$2


if [ -z "$FILENAME" ]; then
  echo "FILENAME is specified, use default."
  FILENAME="all_model.pth";
else
  echo "FILENAME has a value: $FILENAME"
fi

echo "#####################"
echo "#  Model for eval   #"
echo "#####################"

bash download_models.sh $EXP_NAME $FILENAME 

if [ -z "$EXP_NAME" ]; then 
CKPT="None";
else
CKPT=/workspace/model_store/$EXP_NAME/$FILENAME
fi


# name="vit-b-configs/vit_b_32_32_epochs_1_gpu_varyingsize_0_base_0";
name="yolo_configs/yolo_32_epochs_vastai_baseline";
# name="vit-b-configs/vit_b_16_epochs_dist_gpu_varyingsize_0_topk_1";
# name="vit-b-configs/vit_b_16_epochs_1_gpu_varyingsize_0_topk_1";
# name="vit-b-configs/vit_b_16_epochs_1_gpu_varyingsize_1";
# name="vit-b-configs/vit_b_16_epochs_1_gpu_varyingsize_0";
# name="vit-b-configs/vit_b_24_epochs_1_gpu_varyingsize_0";
# name="vit-b-configs/vit_b_24_epochs_1_gpu_0";
# name="rn50_configs/rn50_24_epochs_1_gpu_0";
echo "running "$name;

# echo "running 5layer topk";
# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
python ../../train_load_ckpt.py --config-file ../../$name.yaml \
    --data.train_dataset=/workspace/data/ffcv-data/in1k_train_500_0.50_90.ffcv \
    --data.val_dataset=/workspace/data/ffcv-data/in1k_val_500_0.50_90.ffcv\
    --data.num_workers=32 --data.in_memory=1 \
    --logging.folder=./train_results/$name-original \
    --dist.port 12321 \
    --dist.world_size 2 \
    --resume.model_ckpt $CKPT \
    --resume.resume_model_from_ckpt 1 \
