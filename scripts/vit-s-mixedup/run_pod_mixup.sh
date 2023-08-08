# 8 GPU training (use only 1 for ResNet-18 training)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8;

name="vit-s-configs/vit-s-config_pod";
echo "running "$name;

echo "running non topk";
# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
python ../../train_mixup_randaug.py --config-file ../../$name.yaml \
    --data.train_dataset=/workspace/data/train_500_0.50_90.ffcv \
    --data.val_dataset=/workspace/data/ffcv-imagenet/val_500_0.50_90.ffcv \
    --data.num_workers=24 --data.in_memory=1 \
    --logging.folder=./train_results/$name-original \
    --dist.port 12321 \
    --training.mix_up_alpha 0.2 \

# echo "running topk";
# topk='333366667777';
# # Set the visible GPUs according to the `world_size` configuration parameter
# # Modify `data.in_memory` and `data.num_workers` based on your machine
# python ../../train.py --config-file ../../$name.yaml \
#     --data.train_dataset=/home/ylz1122/data/ffcv-imagenet/train_500_0.50_90.ffcv \
#     --data.val_dataset=/home/ylz1122/data/ffcv-imagenet/val_500_0.50_90.ffcv \
#     --data.num_workers=12 --data.in_memory=1 \
#     --logging.folder=./train_results/$name\_topk\_$topk \
#     --dist.port 12320 \
#     --training.topk_info "$topk" \