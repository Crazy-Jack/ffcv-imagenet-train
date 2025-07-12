import torch
from yolo_v8 import yolo_fpn_nets

# Path to your weights file (absolute path)
weights_path = "/home/i3fellow001/ffcv-imagenet-train/scripts/yolo/train_results/yolo_configs/yolo_32_epochs_vastai_baseline-original/77896d08-b524-4421-b73c-b2819d02fcfa/weights_ep_31.pt"

# Instantiate the model (num_classes=1000 for ImageNet)
model = yolo_fpn_nets.yolo_v8_m(num_classes=1000)

# Load weights
state_dict = torch.load(weights_path, map_location="cpu")

# Remove 'module.' prefix if present
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
print("YOLO FPN model loaded successfully!")
