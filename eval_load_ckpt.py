import torch as ch
from yolo_v8 import yolo_cls_nets, yolo_fpn_nets
import torch
import torchmetrics
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
import os
import torch.multiprocessing as mp
import torch.distributed as dist

Section('model', 'model details').params(
    arch=Param(str, default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)
Section('resume', 'training resume with checkpoints').params(
    model_ckpt=Param(str, 'checkpoint path.pt', default=""),
    resume_model_from_ckpt=Param(int, 'use checkpoint for optimizer?', default=0),
)
Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)
Section('data', 'data related stuff').params(
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)
Section('dist', 'distributed options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def setup_distributed(rank, world_size, address, port):
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def create_val_loader(val_dataset, num_workers, batch_size, resolution, in_memory, distributed):
    res_tuple = (resolution, resolution)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device(f'cuda'), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(f'cuda'), non_blocking=True)
    ]
    loader = Loader(val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    os_cache=in_memory,
                    distributed=distributed)
    return loader

def val_loop(model, val_loader, lr_tta, device):
    from torch.cuda.amp import autocast
    model.eval()
    top1 = torchmetrics.Accuracy(task="multiclass", num_classes=1000).to(device)
    top5 = torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5).to(device)
    mean_loss = torchmetrics.MeanMetric().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        with autocast():
            for images, target in tqdm(val_loader):
                output = model(images)
                if lr_tta:
                    output += model(torch.flip(images, dims=[3]))
                top1(output, target)
                top5(output, target)
                mean_loss(loss_fn(output, target))
    stats = {
        'top_1': top1.compute().item(),
        'top_5': top5.compute().item(),
        'loss': mean_loss.compute().item()
    }
    return stats

@param('model.arch')
@param('model.pretrained')
@param('resume.resume_model_from_ckpt')
@param('resume.model_ckpt')
def load_model(arch, pretrained, resume_model_from_ckpt, model_ckpt, device):
    if arch == 'yolo-v8-m':
        model = yolo_cls_nets.yolo_v8_m(num_classes=1000)
    elif arch == 'yolo-v8-m-fpn':
        model = yolo_fpn_nets.yolo_v8_m(num_classes=1000)
    else:
        from torchvision import models
        model = getattr(models, arch)(pretrained=pretrained)
    if resume_model_from_ckpt:
        print(f"==> Loading model ckpt from {model_ckpt}!")
        checkpoint = torch.load(model_ckpt, map_location="cpu")
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if any(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)
        print("==> Model loaded successfully!")
    else:
        print(f"==> Model created from scratch!")
    model = model.to(device)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    return model

def main_worker(rank, world_size):
    make_config()
    config = get_current_config()
    setup_distributed(rank, world_size, config['dist.address'], config['dist.port']) if world_size > 1 else None
    device = torch.device(f'cuda:{rank}') if world_size > 1 else torch.device('cuda:0')
    model = load_model(device=device)
    val_loader = create_val_loader(
        config['data.val_dataset'],
        config['data.num_workers'],
        config['validation.batch_size'],
        config['validation.resolution'],
        config['data.in_memory'],
        distributed=(world_size > 1)
    )
    stats = val_loop(model, val_loader, config['validation.lr_tta'], device=device)
    if rank == 0 or world_size == 1:
        print("Validation stats:", stats)
    if world_size > 1:
        cleanup_distributed()

def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Load model checkpoint and run evaluation')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    config = get_current_config()
    world_size = config['dist.world_size']
    if world_size > 1:
        mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        main_worker(0, 1)
