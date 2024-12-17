import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from PIL import Image

from torchvision import models
import torch
import torch.nn as nn
import torchmetrics
import numpy as np
import math
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from architecture import *
from pytorch_pretrained_vit import ViT

from torchvision import transforms
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

from yolo_v8 import yolo_cls_nets


Section('model', 'model details').params(
    arch=Param(str, default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic', 'cosine']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1),
    save_model_freq=Param(int, 'save model epoch frequency', default=1)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0),
    topk_info=Param(str, 'topk_info, each digit represent the sparsity level, 0 represent 100%, 1 represent 10%, etc', default=''),
    topk_layer_name=Param(str, 'Topk layer name, specified for which class what to use', default='TopkLayer'),
    alexnet_topk=Param(float, 'alexnet topk, prevent interference', default=0.2),
    resnet50_topk=Param(float, 'resnet50 topk, prevent interference', default=0.2),
    vgg_topk=Param(float, 'VGG topk, prevent interference', default=0.2),
    l1_sparsity_lamda=Param(float, '', default=0),
    topk_tau=Param(float, 'topk tau - the weight that would determine how mauch original activation want to keep, 1 is all and 0 is topk only', default=0.),
    scramble_reverse_weight=Param(float, 'weight of how much reverse optimization would weight', default=1e-3),
    scramble_reverse_lr_scale=Param(float, 'downscale the scamble down scale', default=1e-2),
    topk_decay_ramp=Param(float, 'topk decay ramp?', default=1e+4),
)

Section('resume', 'training resume with checkpoints').params(
    optim_ckpt=Param(str, 'checkpoint path.pt', default=""),
    model_ckpt=Param(str, 'checkpoint path.pt', default=""),
    resume_opt_from_ckpt=Param(int, 'use checkpoint for optimizer?', default=0),
    resume_model_from_ckpt=Param(int, 'use checkpoint for optimizer?', default=0),
    init_eval_checker=Param(int, 'whether to initially check the loaded model', default=0)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cosine_lr(epoch, lr, epochs, lr_peak_epoch):

    eta_min = 0
    lr = eta_min + (lr - eta_min) * (
        1 + math.cos(math.pi * epoch / epochs)) / 2
    return lr

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)




class ImageNetTrainer:
    @param('training.distributed')
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.initialize_logger()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()



    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address

        import socket
        from contextlib import closing

        def find_free_port():
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(('localhost', 0))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return s.getsockname()[1]

        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr,
            'cosine': get_cosine_lr,
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32

        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    @param('resume.resume_opt_from_ckpt')
    @param('resume.optim_ckpt')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, resume_opt_from_ckpt, optim_ckpt):
        assert optimizer == 'sgd'

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        # print(param_groups)
        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        if resume_opt_from_ckpt:
            self.log({'message': f"==> Loading optimizer from ckpt {optim_ckpt}!"})
            self.optimizer.load_state_dict(torch.load(optim_ckpt))
        else:
            self.log({'message': f"==> creating optimizer from scratch!"})

        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
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
                        distributed=distributed)
        return loader

    @param('training.l1_sparsity_lamda')
    def l1_loss_func(self, model, l1_sparsity_lamda):
        if l1_sparsity_lamda > 0:
            l1_loss = []
            for module in model.modules():
                if not isinstance(module, nn.Sequential):
                    if isinstance(module, TopKLayer):
                        l1_loss.append(module.prev_x.unsqueeze(0))
            return torch.cat(l1_loss).sum() * l1_sparsity_lamda
            
    @param('training.epochs')
    @param('logging.log_level')
    @param('logging.save_model_freq')
    @param('resume.init_eval_checker')
    def train(self, epochs, log_level, save_model_freq, init_eval_checker):
        # before begins, check the loaded model if there is one
        if init_eval_checker:
            # check the acc
            self.eval_and_log({'epoch':-1})
            self.best_stats['top_1'] = 0.
            self.best_stats['top_5'] = 0.

        for epoch in range(epochs):
            self.curr_epoch = epoch
            res = self.get_resolution(epoch)
            print(f"Resultion at epoch {epoch} is {res}")
            self.decoder.output_size = (res, res)

            train_loss = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch
                }

                save_or_not = self.eval_and_log(extra_dict)
            
                if save_model_freq > 0 and epoch > 0:
                    if self.gpu == 0 and (epoch % save_model_freq == 0 or epoch == (epochs - 1)):
                        ch.save(self.model.state_dict(), self.log_folder / f'weights_ep_{epoch}.pt')
                        ch.save(self.optimizer.state_dict(), self.log_folder / f'weights_ep_{epoch}_optimizer.pt')
                
                if save_model_freq == -1: # save the best eval models
                    if self.gpu == 0 and save_or_not:
                        print(f"Saving the new best results to {self.log_folder}")
                        ch.save(self.model.state_dict(), self.log_folder / f'weights_best.pt')
                        ch.save(self.optimizer.state_dict(), self.log_folder / f'weights_best_optimizer.pt')
                
        self.eval_and_log({'epoch':epoch})
        # if self.gpu == 0:
        #     ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val

        # determine to save or not
        if stats['top_1'] > self.best_stats['top_1']:
            save_or_not = True 
            self.best_stats['top_1'] = stats['top_1']
            self.best_stats['top_5'] = stats['top_5']
        else:
            save_or_not = False
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'best_top_1': self.best_stats['top_1'],
                'current_top_1': stats['top_1'],
                'current_top_5': stats['top_5'],
                'best_top_5': self.best_stats['top_5'],
                'val_time': val_time
            }, **extra_dict), vis=True)
        
        return save_or_not


    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    @param('training.topk_info')
    @param('training.topk_layer_name')
    @param('resume.resume_model_from_ckpt')
    @param('resume.model_ckpt')
    @param('training.alexnet_topk')
    @param('training.topk_tau')
    @param('training.resnet50_topk')
    @param('training.topk_decay_ramp')
    @param('training.vgg_topk')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool, topk_info, resume_model_from_ckpt, model_ckpt, topk_layer_name, alexnet_topk, resnet50_topk, topk_tau, vgg_topk, topk_decay_ramp):
        scaler = GradScaler()
        if 'vit' == arch.lower()[:3]:
            model_name_vit = arch.split("+")[1] # B_16_imagenet1k


            if model_name_vit == "S":
                print(f"initializing vit-s...")
                # vit small
                model = ViT(pretrained=False,
                            patches=16,
                            dim=384,
                            ff_dim=1536,
                            num_heads=6,
                            num_layers=12,
                            attention_dropout_rate=0.0,
                            dropout_rate=0.1,
                            classifier='token',
                            positional_embedding='1d',
                            image_size=224,
                            topk_layer_name=topk_layer_name,
                            topk_info=topk_info)
            else:
                model = ViT(model_name_vit, pretrained=False, image_size=224, topk_layer_name=topk_layer_name, topk_info=topk_info)
        
        elif 'alexnet_5layers' == arch.lower():
            model = alexnet_5layer(alexnet_topk, pretrained=False, topk_tau=topk_tau)
            print("Using alexnet 5topk layers")
        elif 'alexnet_5layers_finetune' == arch.lower():
            # model = models.alexnet(pretrained=True)
            model = alexnet_5layer(alexnet_topk, pretrained=True, topk_tau=topk_tau)
            print("Using alexnet 5topk layers for finetune")
        elif 'alexnet_5layers_finetune_perm' == arch.lower():
            model = alexnet_5layer(alexnet_topk, pretrained=True, topk_tau=topk_tau, permutate=1)
            print("Using alexnet 5topk layers for finetune")
        elif 'alexnet_5layers_finetune_se' == arch.lower():
            model = alexnet_5layer(alexnet_topk, pretrained=True, topk_tau=topk_tau, take_se_channel=1)
            print("Using alexnet 5topk layers for finetune take_se_channel=1")
        
        elif 'alexnet_5layers_finetune_x3_activation' == arch.lower():
            model = alexnet_5layer(alexnet_topk, pretrained=True, topk_tau=topk_tau, activation='x3')
            print("Using alexnet 5topk layers for activation x3")

        elif 'alexnet_5layers_se' == arch.lower():
            model = alexnet_5layer(alexnet_topk, pretrained=False, topk_tau=topk_tau, take_se_channel=1)
            print("Using alexnet 5topk layers for scratch take_se_channel=1")
        
        elif arch.lower() == 'alexnet_2layer':
            model = alexnet_2layer(alexnet_topk, pretrained=False, topk_tau=topk_tau)
        
        elif arch.lower() == 'alexnet_2layer_finetune':
            model = alexnet_2layer(alexnet_topk, pretrained=True, topk_tau=topk_tau)
        
        elif arch.lower() == 'vgg_5layers':
            model = topK_VGG_5layers(vgg_topk, pretrained=False, topk_tau=topk_tau)
        
        elif arch.lower() == 'vgg_5layers_finetune':
            model = topK_VGG_5layers(vgg_topk, pretrained=True, topk_tau=topk_tau)
        elif arch.lower() == 'ensemblealexnet5layertopk':
            model = EnsembleAlexNet5layerTopK(alexnet_topk)


        elif arch.lower() == 'alexnet':
            alexnet = models.alexnet(pretrained=False)
            model = alexnet

        elif 'resnet50_4layers_finetune' == arch.lower():
            model = topK_resnet50(resnet50_topk, topk_tau=topk_tau, pretrained=True)
        
        elif 'resnet50_4layers_finetune_cosinetopkdecay' == arch.lower():
            model = topK_resnet50(resnet50_topk, topk_tau=topk_tau, pretrained=True, topk_decay_method='cosine', topk_decay_ramp=topk_decay_ramp)

        elif 'resnet50_1layer_finetune' == arch.lower():
            model = topK_resnet50_1layer(resnet50_topk, topk_tau=topk_tau, pretrained=True)

        elif arch == 'yolo-v8-m':
            # introducing yolo-m architecture 
            model = yolo_cls_nets.yolo_v8_m(num_classes=1000)
        else:
            model = getattr(models, arch)(pretrained=pretrained)

        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        if resume_model_from_ckpt:
            self.log({'message': f"==> Loading model ckpt from {model_ckpt}!"})
            if arch in ['yolo-v8-m']:
                checkpoint = torch.load(model_ckpt)['model']
            
            
            else:
                checkpoint = torch.load(model_ckpt)
            
            # editing the mapping keys
            param_count = 0
            new_checkpoint = model.state_dict()
            # print(new_checkpoint.keys())
            for k in checkpoint:
                # print(k)
                new_k = k.replace("module.net.sp_cnn.", "")
                if new_k in new_checkpoint:
                    new_checkpoint[new_k] = checkpoint[k]
                    param_i = 1
                    for i in new_checkpoint[new_k].shape:
                        param_i *= i
                    param_count += param_i
            print(f"==> Total param loaded: {param_count / 1e+6} M")
            model.load_state_dict(new_checkpoint)

        else:
            self.log({'message': f"==> creating model from scratch!"})

        # model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        # model = torch.compile(model)

        return model, scaler

    @param('logging.log_level')
    @param('training.l1_sparsity_lamda')
    def train_loop(self, epoch, log_level, l1_sparsity_lamda):
        model = self.model
        model.train()
        losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                

                output = self.model(images)
                
                loss_train = self.loss(output, target)

                if l1_sparsity_lamda > 0:
                    l1_loss = self.l1_loss_func(self.model)
                    if l1_loss:
                        loss_train += l1_loss


            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end


            ### Logging start
            if log_level > 0:
                if self.gpu == 0:
                    losses.append(loss_train.detach().item())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'lrs']
                values = [epoch, ix, group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']
                    if l1_sparsity_lamda > 0:
                        names += ['l1 loss']
                        values += [f'{l1_loss.item():.3f}']
                        

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)


            ### Logging end
            # if ix > 50:
            #     break

        if len(losses) != 0:
            return sum(losses) / len(losses) * 1.




    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
      
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters['loss'].update(loss_val)
        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task="multiclass", num_classes=1000).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5).to(self.gpu),
            'loss': torchmetrics.aggregation.MeanMetric().to(self.gpu)
        }

        self.best_stats = {
            'top_1': 0.,
            'top_5': 0.,
            'loss': 0.
        }

        if self.gpu == 0:
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

    def val_image_feature_maps(self, dir, cur_time):
        
        img_ = Image.open("/home/ylz1122/ffcv-imagenet-train/airplane1-chair2.png")
        tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
        img = tfms(img_).unsqueeze(0)
        img = img.to("cuda:0")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img)
        
        layer_sparse_activation = {}
        for name, m in self.model.module.features.named_children():
            if isinstance(m, TopKLayer):
                layer_sparse_activation[name] = m.sparse_x.detach().cpu()
        target_tensor = layer_sparse_activation['3']
        target_tensor = target_tensor.squeeze(0).unsqueeze(1).repeat(1, 3, 1, 1)#.mean(0, keepdim=True)
        img_size = (img.shape[-2], img.shape[-1])
        target_tensor = torch.nn.functional.interpolate(target_tensor, size=img_size)
        n, c, h, w = target_tensor.shape
        target_tensor = target_tensor / target_tensor.max()
        target_tensor = target_tensor.add(1).mul(0.5)
        grid_img = torchvision.utils.make_grid(target_tensor, nrow=5)
        plt.clf()
        plt.figure(figsize=(50, 50))
        # plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        os.makedirs(os.path.join(dir, "vis"), exist_ok=True)
        plt.savefig(os.path.join(dir, "vis", f"vis_layer_3_{cur_time}.png"))
        print(f"==> Log visualization in {os.path.join(dir, 'vis')}")
        
        
    def log(self, content, vis=False):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()
        
        # handle the img log
        # if vis:
        #     self.val_image_feature_maps(self.log_folder, cur_time)
        

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":

    make_config()
    ImageNetTrainer.launch_from_args()
