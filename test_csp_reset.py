#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

# Import the CSP_Reset class
from yolo_v8.yolo_fpn_nets_topk_v1 import CSP_Reset

def test_csp_reset_basic():
    """Test basic functionality of CSP_Reset"""
    print("=== Testing CSP_Reset Basic Functionality ===\n")
    
    # Create CSP_Reset module
    in_channels = 64
    out_channels = 128
    csp_reset = CSP_Reset(in_channels, out_channels, n=2)
    
    print(f"CSP_Reset created with {in_channels} input channels and {out_channels} output channels")
    
    # Create dummy input
    batch_size = 2
    height, width = 16, 16
    x = torch.randn(batch_size, in_channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    print("\n--- Forward Pass ---")
    output = csp_reset(x)
    print(f"Output shape: {output.shape}")
    
    # Check channel usage stats
    stats = csp_reset.get_channel_usage_stats()
    print(f"\nChannel Usage Stats:")
    print(f"  Total channels: {stats['total_channels']}")
    print(f"  Used channels: {stats['used_channels']}")
    print(f"  Usage ratio: {stats['usage_ratio']:.4f}")
    print(f"  Used channel indices: {stats['used_channel_indices'][:10]}...")  # Show first 10
    
    return csp_reset, x

def test_channel_reset():
    """Test different reset methods"""
    print("\n=== Testing Channel Reset Methods ===\n")
    
    csp_reset, x = test_csp_reset_basic()
    
    # Get initial weights for comparison
    initial_weights = csp_reset.conv3.conv.weight.clone()
    
    # Test random reset
    print("\n--- Testing Random Reset ---")
    csp_reset.reset_used_channels(reset_type='random')
    
    # Check if weights changed
    weight_diff = torch.norm(csp_reset.conv3.conv.weight - initial_weights)
    print(f"Weight change after random reset: {weight_diff:.6f}")
    
    # Forward pass again to accumulate new used channels
    output = csp_reset(x)
    stats = csp_reset.get_channel_usage_stats()
    print(f"Used channels after reset: {stats['used_channels']}")
    
    # Test original reset
    print("\n--- Testing Original Reset ---")
    csp_reset.reset_used_channels(reset_type='original')
    
    # Check if weights are back to original
    weight_diff = torch.norm(csp_reset.conv3.conv.weight - initial_weights)
    print(f"Weight difference from original: {weight_diff:.6f}")
    
    # Test zero reset
    print("\n--- Testing Zero Reset ---")
    csp_reset.reset_used_channels(reset_type='zero')
    
    # Check if weights are zero
    zero_norm = torch.norm(csp_reset.conv3.conv.weight)
    print(f"Norm of weights after zero reset: {zero_norm:.6f}")

def test_training_vs_eval():
    """Test channel tracking in training vs evaluation mode"""
    print("\n=== Testing Training vs Evaluation Mode ===\n")
    
    csp_reset, x = test_csp_reset_basic()
    
    # Clear any existing tracking
    csp_reset.enable_channel_tracking(True)
    csp_reset.used_channels.clear()
    csp_reset.channel_usage_count.zero_()
    
    # Test in training mode
    print("--- Training Mode ---")
    csp_reset.train()
    output = csp_reset(x)
    stats = csp_reset.get_channel_usage_stats()
    print(f"Used channels in training: {stats['used_channels']}")
    
    # Test in evaluation mode
    print("\n--- Evaluation Mode ---")
    csp_reset.eval()
    output = csp_reset(x)
    stats = csp_reset.get_channel_usage_stats()
    print(f"Used channels in eval: {stats['used_channels']}")
    
    # Note: In eval mode, tracking should be disabled by default
    print("Note: Channel tracking is typically disabled in eval mode for efficiency")

def test_usage_threshold():
    """Test different usage thresholds"""
    print("\n=== Testing Usage Thresholds ===\n")
    
    csp_reset, x = test_csp_reset_basic()
    
    # Modify the threshold in the forward method for testing
    original_threshold = 1e-6
    
    # Test with different thresholds
    thresholds = [1e-8, 1e-6, 1e-4, 1e-2]
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold} ---")
        
        # Clear tracking
        csp_reset.used_channels.clear()
        csp_reset.channel_usage_count.zero_()
        
        # Forward pass with modified threshold
        output = csp_reset(x)
        
        # Manually check with the threshold
        with torch.no_grad():
            channel_activations = output.abs().mean(dim=[0, 2, 3])
            used_mask = channel_activations > threshold
            used_count = used_mask.sum().item()
        
        print(f"Channels with activation > {threshold}: {used_count}/{output.shape[1]}")
        print(f"Usage ratio: {used_count/output.shape[1]:.4f}")

def test_multiple_forward_passes():
    """Test channel usage across multiple forward passes"""
    print("\n=== Testing Multiple Forward Passes ===\n")
    
    csp_reset, x = test_csp_reset_basic()
    
    # Clear tracking
    csp_reset.used_channels.clear()
    csp_reset.channel_usage_count.zero_()
    
    # Multiple forward passes
    num_passes = 5
    for i in range(num_passes):
        output = csp_reset(x)
        stats = csp_reset.get_channel_usage_stats()
        print(f"Pass {i+1}: Used channels = {stats['used_channels']}, Usage ratio = {stats['usage_ratio']:.4f}")
    
    # Show usage count distribution
    usage_count = csp_reset.channel_usage_count
    print(f"\nUsage count statistics:")
    print(f"  Mean usage count: {usage_count.float().mean():.2f}")
    print(f"  Max usage count: {usage_count.max()}")
    print(f"  Min usage count: {usage_count.min()}")
    print(f"  Channels used in all passes: {(usage_count == num_passes).sum()}")

def test_reset_all_channels():
    """Test resetting all channels"""
    print("\n=== Testing Reset All Channels ===\n")
    
    csp_reset, x = test_csp_reset_basic()
    
    # Get initial weights
    initial_weights = csp_reset.conv3.conv.weight.clone()
    
    # Reset all channels
    print("--- Resetting All Channels ---")
    csp_reset.reset_all_channels(reset_type='random')
    
    # Check if all weights changed
    weight_diff = torch.norm(csp_reset.conv3.conv.weight - initial_weights)
    print(f"Weight change after resetting all channels: {weight_diff:.6f}")
    
    # Verify all channels are marked as used
    stats = csp_reset.get_channel_usage_stats()
    print(f"All channels marked as used: {stats['used_channels'] == stats['total_channels']}")

if __name__ == "__main__":
    print("CSP_Reset Channel Tracking and Reset Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_csp_reset_basic()
        test_channel_reset()
        test_training_vs_eval()
        test_usage_threshold()
        test_multiple_forward_passes()
        test_reset_all_channels()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc() 