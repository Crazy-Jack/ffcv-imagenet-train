import math

import torch

# from utils.util import make_anchors

def sparse_hw(x, topk):
    
    n, c, h, w = x.shape
    if topk == 1:
        return x
    x_reshape = x.view(n, c, h * w)
    topk_keep_num = int(max(1, topk * h * w))
    _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
    mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(x.device)
    
    sparse_x = mask * x_reshape
    
    return sparse_x.view(n, c, h, w)


def compute_spatial_std(binary_tensor):
    """
    Compute standard deviation of spatial coordinates (x,y) of non-zero elements.
    
    Args:
        binary_tensor: Tensor of shape [b, c, h, w] with binary values (0 or 1)
    
    Returns:
        Tensor of shape [b, c] containing standard deviation of x,y coordinates
        for each batch and channel
    """
    b, c, h, w = binary_tensor.shape
    
    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=binary_tensor.device, dtype=binary_tensor.dtype),
                                       torch.arange(w, device=binary_tensor.device, dtype=binary_tensor.dtype),
                                       indexing='ij')
    
    # Expand coordinates to match tensor dimensions
    y_coords = y_coords.unsqueeze(0).unsqueeze(0).expand(b, c, h, w)  # [b, c, h, w]
    x_coords = x_coords.unsqueeze(0).unsqueeze(0).expand(b, c, h, w)  # [b, c, h, w]
    
    # Get non-zero positions
    non_zero_mask = binary_tensor > 0  # [b, c, h, w]

    y_one_coords = (non_zero_mask * y_coords).reshape(b, c, -1)
    x_one_coords = (non_zero_mask * x_coords).reshape(b, c, -1)
    return 0.5 * (y_one_coords.std(dim=2) + x_one_coords.std(dim=2))
    
   
    

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class CSP_Reset(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True, topk=1., keep_ratio=0.5):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))
        
        # Track used channels - register as buffer to work with DataParallel/DistributedDataParallel
        self.register_buffer('channel_usage_count', torch.zeros(out_ch))
        self.track_channels = True
        
        # Store original initialization for reset
        self.original_conv3_weight = self.conv3.conv.weight.clone().detach()
        self.original_conv3_bias = self.conv3.conv.bias.clone().detach() if self.conv3.conv.bias is not None else None
        self.topk = topk
        self.keep_ratio = keep_ratio
        self.num_keep = int(self.keep_ratio * out_ch)

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        concatenated = torch.cat(y, dim=1)
        
        # Forward through conv3
        conv3_output = self.conv3(concatenated)
        conv3_output = sparse_hw(conv3_output, self.topk)
        
        # Create binary mask from sparse output
        binary_mask = (conv3_output != 0).float()
        
        # Compute spatial standard deviation
        spatial_std = compute_spatial_std(binary_mask) # [b, c]
        
        # Track which channels are being used (have non-zero activations)
        if self.track_channels and self.training:
            with torch.no_grad():
                # Reduce from [b, c] to [c] by taking mean across batches
                mean_spatial_std = spatial_std.mean(dim=0)  # [c]
                
                # Sort channels by spatial std (ascending order - smaller values first)
                sorted_indices = torch.argsort(mean_spatial_std)
                
                # Get indices of top-k channels
                topk_indices = sorted_indices[:self.num_keep]
                
                # Update usage count - this will work across GPUs with DataParallel
                self.channel_usage_count[topk_indices] += 1

        return conv3_output
    
    def reset_used_channels(self):
        """
        Reset the weights of channels that have not been used.
        This method should be called after synchronization across GPUs.
        """
        # if self.channel_usage_count.sum() == 0:
        #     print("No channels to reset - no channels have been used yet.")
        #     return

        sorted_indices = torch.argsort(self.channel_usage_count, descending=True)

        # Get indices of unfit channels
        indices = sorted_indices[self.num_keep:]
        reset_indices = indices.cpu().numpy()
        
        print(f"Resetting {len(reset_indices)} channels: {reset_indices.tolist()}")
        
        with torch.no_grad():
            # Reset to original initialization
            for ch_idx in reset_indices:
                self.conv3.conv.weight[ch_idx] = self.original_conv3_weight[ch_idx]
                if self.conv3.conv.bias is not None and self.original_conv3_bias is not None:
                    self.conv3.conv.bias[ch_idx] = self.original_conv3_bias[ch_idx]
        
        # Clear the usage count after reset
        self.channel_usage_count.zero_()
        
        print(f"Successfully reset {len(reset_indices)} channels using original weights.")
    
    def synchronize_usage_counts(self):
        """
        Synchronize usage counts across all GPUs.
        This should be called before reset_used_channels in multi-GPU training.
        """
        if torch.distributed.is_initialized():
            # All-reduce the usage counts across all processes
            torch.distributed.all_reduce(self.channel_usage_count, op=torch.distributed.ReduceOp.SUM)
        elif hasattr(self, 'module'):  # DataParallel case
            # In DataParallel, the buffer is automatically synchronized
            pass


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, topk_info):
        super().__init__()
        self.topk_info = topk_info
        # parse topk_info : "topk_layer_name:topk"
        self.topk_info = {k: float(v) for k, v in [info.split(':') for info in topk_info.split(',')]}
        print(f"---> top-k info: {self.topk_info}  <---")

        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP_Reset(width[3], width[3], depth[1], topk=self.topk_info["p3"])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP_Reset(width[4], width[4], depth[2], topk=self.topk_info["p4"])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP_Reset(width[5], width[5], depth[0], topk=self.topk_info["p5"]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))

        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3),
                                                           Conv(c1, c1, 3),
                                                           torch.nn.Conv2d(c1, self.nc, 1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3),
                                                           Conv(c2, c2, 3),
                                                           torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
        if self.training:
            return x
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, num_classes, topk_info):
        super().__init__()
        

        self.net = DarkNet(width, depth, topk_info)
        self.fpn = DarkFPN(width, depth)

        
        # img_dummy = torch.zeros(1, 3, 256, 256)
        # self.head = Head(num_classes, (width[3], width[4], width[5]))
        # self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        # self.stride = self.head.stride
        # self.head.initialize_biases()

        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        img_dummy = torch.zeros(1, 3, 160, 160)
        _, _, p5 = self.fpn(self.net(img_dummy))
        
        self.fc = torch.nn.Linear(p5.shape[1], num_classes)

    def forward(self, x, output_intermediate=False):
        p3, p4, p5 = self.net(x)
        
        
        x = self.fpn([p3, p4, p5])
        p3, p4, p5 = x
        feature = self.adaptive_avg_pool(p5).squeeze(-1).squeeze(-1)
        return self.fc(feature)
        

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
    
    def reset_weights(self):
        for m in self.modules():
            if type(m) is CSP_Reset:
                m.synchronize_usage_counts()
                m.reset_used_channels()


def yolo_v8_n(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes)


def yolo_v8_s(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_m(num_classes: int = 80, topk_info=None):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes, topk_info)


def yolo_v8_l(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_x(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth, num_classes)

if __name__ == "__main__":
    model = yolo_v8_m(80).cuda()
    print(model)
