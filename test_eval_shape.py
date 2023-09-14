
from ShapeBiasEval import run_evaluation
from .train import create_model_and_scaler

import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


modelnames="anyModel"
model_ckpt = f"/home/ylz1122/ffcv-imagenet-train/scripts/alexnet/train_results/alexnet_configs/alexnet_5layers_32_epochs_test_finetune_alex_2layer-original/06eec4be-88e5-43ef-8f52-6d9f827d8805/weights_ep_{ep}.pt"
ep = 5

vgg16 = create_model_and_scaler(
    
)
# 1. evaluate models on out-of-distribution datasets
result = run_evaluation(modelnames,vgg16)
print(result)
