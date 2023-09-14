"""
This script tries to answer the question of whether topk trained network can produce more coherent / continuous parts representation
"""

from architecture import *
import torch.nn as nn 
import torch
import torchvision.models as models
import json
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import torch
from torchvision import models
from torchvision import transforms


import requests
from io import BytesIO

class BlurPoolConv2d(nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)


def apply_blurpool(mod: nn.Module):
    for (name, child) in mod.named_children():
        if isinstance(child, nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
            setattr(mod, name, BlurPoolConv2d(child))
        else: apply_blurpool(child)


def topK_AlexNet(pretrain_weigth, topk, tau=0., perm=0, blurpool=0, take_se_channel=0, **kwargs):
    if pretrain_weigth=="":
        alexnet = alexnet_5layer(topk, pretrained=True, topk_tau=tau, permutate=perm, take_se_channel=take_se_channel)
    else:
        alexnet = alexnet_5layer(topk, pretrained=False, topk_tau=tau, permutate=perm, take_se_channel=take_se_channel)
    
    if blurpool:
        apply_blurpool(alexnet)

    if pretrain_weigth:
        # print("alexnet model dict"[])
        # print([i for i in alexnet.state_dict()])
        # print("load dict")
        # print([i for i in torch.load(pretrain_weigth)])
        pretrain_weigth += "weights_best.pt"
        ckpt = torch.load(pretrain_weigth)
        new_dict = {}
        for k in ckpt:
            new_dict[k.replace("module.", "")] = ckpt[k]
        alexnet.load_state_dict(new_dict)
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # alexnet = alexnet.to(device)

    return alexnet


topk = 0.2
topk_tau = 0.
pretrained_weight = "/home/ylz1122/ffcv-imagenet-train/scripts/alexnet/train_results/alexnet_configs/alexnet_5layers_finetune_2/b86f3b97-95f4-45e7-b1d5-9c84b38c3280/"
# pretrained_weight = ""
model = topK_AlexNet(pretrained_weight, topk, tau=topk_tau, perm=0, blurpool=0, take_se_channel=0)

# data


url = "https://raw.githubusercontent.com/rgeirhos/texture-vs-shape/master/stimuli/style-transfer-preprocessed-512/airplane/airplane1-chair2.png"
# url = "https://raw.githubusercontent.com/rgeirhos/texture-vs-shape/master/stimuli/style-transfer-preprocessed-512/dog/dog10-elephant1.png"
# url = "https://raw.githubusercontent.com/rgeirhos/texture-vs-shape/master/stimuli/style-transfer-preprocessed-512/car/car10-chair1.png"
response = requests.get(url)
img_ = Image.open(BytesIO(response.content))
# img_ = Image.open("/home/ylz1122/nips2023_shape_vs_texture/topk-neurons-visualization-supp/shapebiasbench/airplane1-chair2.png")
# img_= Image.open("/home/ylz1122/data/few-shot/airline/n02690373_1381.JPEG")

# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
img = tfms(img_).unsqueeze(0)

# Load class names
import json
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

model.eval()
with torch.no_grad():
    outputs = model(img).squeeze(0)
print('-----')
for idx in torch.topk(outputs, k=10).indices.tolist():
    prob = torch.softmax(outputs, -1)[idx].item()
    print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))
    
layer_sparse_activation = {}
for name, m in model.features.named_children():
  if isinstance(m, TopKLayer):
    print(m.sparse_x.shape)
    layer_sparse_activation[name] = m.sparse_x ** 3




blend_rate = 1
target_tensor = layer_sparse_activation['3']
target_tensor = target_tensor.squeeze(0).unsqueeze(1).repeat(1, 3, 1, 1)#.mean(0, keepdim=True)

img_size = (img.shape[-2], img.shape[-1])
target_tensor = torch.nn.functional.interpolate(target_tensor, size=img_size)

n, c, h, w = target_tensor.shape
# # target_tensor = torch.softmax(target_tensor.reshape(n, c, h * w), 2).reshape(n, c, h, w)
print(target_tensor.shape)
# # target_tensor = (target_tensor * img )
target_tensor = target_tensor / target_tensor.max()
target_tensor = target_tensor.add(1).mul(0.5)
print(target_tensor.device)
# print(target_tensor.shape)
# print(img.shape)
# # make grid (2 rows and 5 columns) to display our 10 images
grid_img = torchvision.utils.make_grid(target_tensor, nrow=5)
plt.figure(figsize=(40, 40))
# plt.figure(figsize=(10, 10))
plt.imshow(grid_img.permute(1, 2, 0))
# plt.savefig("alex_vis_original.png")
title = f"qu_alex_vis_img_2"
if pretrained_weight != "":
    title = f"{title}_finetune_"

title = f"{title}_{topk}"
plt.savefig(f"{title}.png")