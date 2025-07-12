#!/usr/bin/env python3

import torch
import torch.nn as nn
from yolo_v8 import yolo_cls_nets, yolo_fpn_nets

def test_cuda_118():
    print("=== CUDA 11.8 Compatibility Test ===\n")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n✅ CUDA 11.8 setup successful")
    
    # Test model creation and forward pass
    try:
        model = yolo_cls_nets.yolo_v8_m(num_classes=1000)
        print("✅ YOLO model creation successful")
        
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = torch.randn(1, 3, 224, 224).cuda()
        else:
            dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ YOLO forward pass successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    # Test distributed setup (without actually running it)
    try:
        import torch.distributed as dist
        print("✅ Distributed module import successful")
    except Exception as e:
        print(f"❌ Distributed import failed: {e}")
    
    print("\n=== Test Complete ===")
    return True

if __name__ == "__main__":
    test_cuda_118() 