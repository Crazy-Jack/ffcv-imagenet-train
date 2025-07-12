# conda activate ffcv
export ENV_NAME=ffcv && \
export WORK_DIR=/home/i3fellow001
export WORK_ENV_DIR=/home/i3fellow001/env
export ENV_NAME=$WORK_ENV_DIR/miniconda/envs/ffcv && \
# source activate $ENV_NAME && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib && \
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$WORK_ENV_DIR/Install-OpenCV/source/lib/pkgconfig && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORK_ENV_DIR/Install-OpenCV/source/lib && \
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$WORK_ENV_DIR/Install-libjpeg-turbo/install/lib/pkgconfig && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORK_ENV_DIR/Install-libjpeg-turbo/install/lib/ && \

export CUDA_VISIBLE_DEVICES=0,1;

python3 ../../eval_load_ckpt.py \
  --model.arch yolo-v8-m \
  --resume.resume_model_from_ckpt 1 \
  --resume.model_ckpt /home/i3fellow001/ffcv-imagenet-train/scripts/yolo/train_results/yolo_configs/yolo_32_epochs_vastai_baseline-original/77896d08-b524-4421-b73c-b2819d02fcfa/weights_ep_31.pt \
  --data.val_dataset /home/i3fellow001/data/ffcv-data/in1k_val_500_0.50_90.ffcv \
  --data.num_workers 4 \
  --data.in_memory 1 \
  --dist.port 12323 \
  --dist.world_size 2 \
  --dist.address localhost \
  --validation.batch_size 32 \
  --validation.resolution 224 \
  --validation.lr_tta 0