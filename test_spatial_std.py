import torch
import numpy as np
from yolo_v8.yolo_fpn_nets_topk_v1 import CSP_Reset, compute_spatial_std

def test_spatial_std_function():
    """Test the compute_spatial_std function with various scenarios"""
    print("Testing compute_spatial_std function...")
    
    # Test 1: Simple case with known pattern
    print("\n1. Testing simple 2x2 pattern:")
    binary_tensor = torch.tensor([
        [[[1, 0],
          [0, 1]]]
    ], dtype=torch.float32)  # [1, 1, 2, 2]
    
    spatial_std = compute_spatial_std(binary_tensor)
    print(f"Input shape: {binary_tensor.shape}")
    print(f"Binary tensor:\n{binary_tensor[0, 0]}")
    print(f"Spatial std: {spatial_std}")
    
    # Test 2: Multiple channels and batches
    print("\n2. Testing multiple channels and batches:")
    binary_tensor = torch.tensor([
        [[[1, 0, 1],
          [0, 1, 0],
          [1, 0, 1]]],
        [[[0, 1, 0],
          [1, 0, 1],
          [0, 1, 0]]]
    ], dtype=torch.float32)  # [2, 1, 3, 3]
    
    spatial_std = compute_spatial_std(binary_tensor)
    print(f"Input shape: {binary_tensor.shape}")
    print(f"Batch 0:\n{binary_tensor[0, 0]}")
    print(f"Batch 1:\n{binary_tensor[1, 0]}")
    print(f"Spatial std: {spatial_std}")
    
    # Test 3: All zeros case
    print("\n3. Testing all zeros case:")
    binary_tensor = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
    spatial_std = compute_spatial_std(binary_tensor)
    print(f"Input shape: {binary_tensor.shape}")
    print(f"Spatial std: {spatial_std}")
    
    # Test 4: Single non-zero element
    print("\n4. Testing single non-zero element:")
    binary_tensor = torch.zeros(1, 1, 5, 5, dtype=torch.float32)
    binary_tensor[0, 0, 2, 3] = 1  # Single element at position (2, 3)
    spatial_std = compute_spatial_std(binary_tensor)
    print(f"Input shape: {binary_tensor.shape}")
    print(f"Non-zero position: (2, 3)")
    print(f"Spatial std: {spatial_std}")
    
    # Test 5: Random sparse pattern
    print("\n5. Testing random sparse pattern:")
    torch.manual_seed(42)
    binary_tensor = torch.zeros(2, 3, 8, 8, dtype=torch.float32)
    # Add random non-zero elements
    for b in range(2):
        for c in range(3):
            num_nonzero = torch.randint(5, 15, (1,)).item()
            indices = torch.randperm(64)[:num_nonzero]
            for idx in indices:
                h, w = idx // 8, idx % 8
                binary_tensor[b, c, h, w] = 1
    
    spatial_std = compute_spatial_std(binary_tensor)
    print(f"Input shape: {binary_tensor.shape}")
    print(f"Non-zero counts per batch/channel:")
    for b in range(2):
        for c in range(3):
            count = binary_tensor[b, c].sum().item()
            std_val = spatial_std[b, c].item()
            print(f"  Batch {b}, Channel {c}: {count} non-zero, std={std_val:.3f}")

def test_csp_reset_with_spatial_std():
    """Test CSP_Reset with spatial standard deviation computation"""
    print("\n" + "="*50)
    print("Testing CSP_Reset with spatial standard deviation...")
    
    # Create CSP_Reset module
    csp = CSP_Reset(in_ch=64, out_ch=128, n=1, topk=0.3)
    
    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 64, 16, 16)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"CSP output channels: {csp.conv3.conv.out_channels}")
    
    # Test forward pass without spatial std
    print("\n1. Forward pass without spatial std:")
    output = csp(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Non-zero elements: {(output != 0).sum().item()}")
    
    # Test forward pass with spatial std
    print("\n2. Forward pass with spatial std:")
    output, spatial_std = csp(input_tensor, return_spatial_std=True)
    print(f"Output shape: {output.shape}")
    print(f"Spatial std shape: {spatial_std.shape}")
    print(f"Non-zero elements: {(output != 0).sum().item()}")
    
    # Print spatial std statistics
    print(f"\nSpatial std statistics:")
    print(f"  Mean: {spatial_std.mean().item():.4f}")
    print(f"  Std: {spatial_std.std().item():.4f}")
    print(f"  Min: {spatial_std.min().item():.4f}")
    print(f"  Max: {spatial_std.max().item():.4f}")
    
    # Show some examples
    print(f"\nSample spatial std values:")
    for b in range(min(2, batch_size)):
        for c in range(min(5, spatial_std.shape[1])):
            std_val = spatial_std[b, c].item()
            nonzero_count = (output[b, c] != 0).sum().item()
            print(f"  Batch {b}, Channel {c}: std={std_val:.3f}, non-zero={nonzero_count}")

def test_spatial_std_interpretation():
    """Test and explain what the spatial std means"""
    print("\n" + "="*50)
    print("Understanding spatial standard deviation...")
    
    # Create different patterns to demonstrate spatial std meaning
    patterns = {
        "Concentrated": torch.zeros(1, 1, 8, 8),
        "Spread": torch.zeros(1, 1, 8, 8),
        "Corner": torch.zeros(1, 1, 8, 8),
        "Center": torch.zeros(1, 1, 8, 8),
    }
    
    # Concentrated pattern (low std)
    patterns["Concentrated"][0, 0, 3:5, 3:5] = 1
    
    # Spread pattern (high std)
    patterns["Spread"][0, 0, 0, 0] = 1
    patterns["Spread"][0, 0, 0, 7] = 1
    patterns["Spread"][0, 0, 7, 0] = 1
    patterns["Spread"][0, 0, 7, 7] = 1
    
    # Corner pattern (medium std)
    patterns["Corner"][0, 0, 0:2, 0:2] = 1
    
    # Center pattern (low std)
    patterns["Center"][0, 0, 3, 3] = 1
    patterns["Center"][0, 0, 3, 4] = 1
    patterns["Center"][0, 0, 4, 3] = 1
    patterns["Center"][0, 0, 4, 4] = 1
    
    print("\nPattern analysis:")
    for name, pattern in patterns.items():
        spatial_std = compute_spatial_std(pattern)
        std_val = spatial_std[0, 0].item()
        nonzero_count = pattern.sum().item()
        
        print(f"\n{name} pattern:")
        print(f"  Non-zero elements: {nonzero_count}")
        print(f"  Spatial std: {std_val:.3f}")
        print(f"  Pattern visualization:")
        for i in range(8):
            row = "  "
            for j in range(8):
                if pattern[0, 0, i, j] > 0:
                    row += "1 "
                else:
                    row += "0 "
            print(row)

if __name__ == "__main__":
    test_spatial_std_function()
    test_csp_reset_with_spatial_std()
    test_spatial_std_interpretation()
    
    print("\n" + "="*50)
    print("All tests completed!") 