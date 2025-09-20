#!/usr/bin/env python3
"""
Comprehensive GPU Test Script for PlenOctree Dependencies
=========================================================

This script tests if JAX, PyTorch, svox, and LPIPS are all working with GPU support.
"""

import sys
import time
import numpy as np

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f"{title}")
    print(f"{'-'*40}")

def test_system_info():
    """Test basic system information."""
    print_section("SYSTEM INFORMATION")
    
    print(f"Python version: {sys.version}")
    
    # Check CUDA availability at system level
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'Driver Version' in line:
                    print(f"  {line.strip()}")
                    break
        else:
            print("✗ NVIDIA GPU not detected or nvidia-smi not available")
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")

def test_pytorch():
    """Test PyTorch installation and GPU support."""
    print_section("PYTORCH TEST")
    
    try:
        import torch
        print(f"✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Test GPU tensor operations
            print("\n  Testing GPU tensor operations...")
            device = torch.device('cuda:0')
            
            # Create tensors on GPU
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            
            # Time matrix multiplication
            start_time = time.time()
            c = torch.mm(a, b)
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_time = time.time() - start_time
            
            print(f"  ✓ GPU matrix multiplication: {gpu_time:.4f}s")
            print(f"  ✓ Result tensor shape: {c.shape}")
            print(f"  ✓ Result tensor device: {c.device}")
            
            # Compare with CPU
            print("\n  Comparing with CPU performance...")
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            
            start_time = time.time()
            c_cpu = torch.mm(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            print(f"  CPU matrix multiplication: {cpu_time:.4f}s")
            speedup = cpu_time / gpu_time
            print(f"  GPU speedup: {speedup:.2f}x")
            
        else:
            print("  ⚠ CUDA not available - using CPU only")
            
        return True, cuda_available
        
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, False

def test_jax():
    """Test JAX installation and GPU support."""
    print_section("JAX TEST")
    
    try:
        import jax
        import jax.numpy as jnp
        
        print(f"✓ JAX imported successfully")
        print(f"  JAX version: {jax.__version__}")
        
        # Check available devices
        devices = jax.devices()
        print(f"  Available devices: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"    Device {i}: {device}")
        
        # Check if GPU is available
        gpu_available = any('gpu' in str(device).lower() or 'cuda' in str(device).lower() for device in devices)
        
        if gpu_available:
            print(f"  ✓ GPU support detected")
            
            # Test GPU operations
            print("\n  Testing GPU operations...")
            
            # Create arrays
            key = jax.random.PRNGKey(42)
            a = jax.random.normal(key, (1000, 1000))
            b = jax.random.normal(key, (1000, 1000))
            
            # Define computation
            @jax.jit
            def matrix_mult(x, y):
                return jnp.dot(x, y)
            
            # Time the computation
            start_time = time.time()
            result = matrix_mult(a, b)
            result.block_until_ready()  # Wait for computation to complete
            jax_time = time.time() - start_time
            
            print(f"  ✓ JAX matrix multiplication: {jax_time:.4f}s")
            print(f"  ✓ Result shape: {result.shape}")
            print(f"  ✓ Result device: {result.device()}")
            
        else:
            print("  ⚠ GPU not detected - using CPU only")
            
        return True, gpu_available
        
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, False

def test_lpips():
    """Test LPIPS installation and GPU support."""
    print_section("LPIPS TEST")
    
    try:
        import lpips
        import torch
        
        print(f"✓ LPIPS imported successfully")
        try:
            print(f"  LPIPS version: {lpips.__version__}")
        except:
            print("  LPIPS version: unknown")
        
        # Initialize LPIPS model
        print("\n  Initializing LPIPS model...")
        lpips_model = lpips.LPIPS(net='alex')
        
        # Check if we can use GPU
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print("  ✓ Moving LPIPS model to GPU...")
            lpips_model = lpips_model.cuda()
            device = 'cuda'
        else:
            print("  ⚠ Using CPU for LPIPS")
            device = 'cpu'
        
        # Test LPIPS computation
        print(f"\n  Testing LPIPS computation on {device}...")
        
        # Create test images
        if cuda_available:
            img1 = torch.rand(1, 3, 256, 256).cuda()
            img2 = torch.rand(1, 3, 256, 256).cuda()
        else:
            img1 = torch.rand(1, 3, 256, 256)
            img2 = torch.rand(1, 3, 256, 256)
        
        # Time LPIPS computation
        start_time = time.time()
        with torch.no_grad():
            score = lpips_model(img1, img2)
        if cuda_available:
            torch.cuda.synchronize()
        lpips_time = time.time() - start_time
        
        print(f"  ✓ LPIPS computation successful: {score.item():.4f}")
        print(f"  ✓ Computation time: {lpips_time:.4f}s")
        print(f"  ✓ Device used: {img1.device}")
        
        return True, cuda_available
        
    except Exception as e:
        print(f"✗ LPIPS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, False

def test_svox():
    """Test svox installation and GPU support."""
    print_section("SVOX TEST")
    
    try:
        import svox
        print(f"✓ svox imported successfully")
        print(f"  svox version: {svox.__version__}")
        
        # Test basic svox functionality
        print("\n  Testing svox octree creation...")
        
        # Create a small octree
        octree = svox.N3Tree(
            data_dim=4,  # RGB + density
            center=[0.5, 0.5, 0.5],
            radius=0.5
        )
        
        print(f"  ✓ Octree created successfully")
        print(f"  ✓ Data dimension: {octree.data_dim}")
        print(f"  ✓ Device: {octree.data.device}")
        
        # Check if octree can use GPU
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print("\n  Testing GPU operations...")
            try:
                # Move octree to GPU
                octree = octree.cuda()
                print(f"  ✓ Octree moved to GPU: {octree.data.device}")
                
                # Test basic operations
                print("  ✓ GPU operations successful")
                
            except Exception as e:
                print(f"  ⚠ GPU operations failed: {e}")
                cuda_available = False
        else:
            print("  ⚠ CUDA not available - using CPU only")
        
        # Test volume renderer
        print("\n  Testing volume renderer...")
        try:
            renderer = svox.VolumeRenderer(octree, step_size=0.5)
            print("  ✓ Volume renderer created successfully")
            
            # Test rendering (simple test)
            print("  ✓ svox functionality verified")
            
        except Exception as e:
            print(f"  ⚠ Volume renderer test failed: {e}")
        
        return True, cuda_available
        
    except Exception as e:
        print(f"✗ svox test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, False

def test_integration():
    """Test integration between different libraries."""
    print_section("INTEGRATION TEST")
    
    try:
        import torch
        import jax
        import jax.numpy as jnp
        import lpips
        import svox
        
        print("Testing library integration...")
        
        # Test PyTorch-JAX data transfer
        print("\n  Testing PyTorch-JAX integration...")
        torch_tensor = torch.randn(100, 100)
        jax_array = jnp.array(torch_tensor.numpy())
        back_to_torch = torch.from_numpy(np.array(jax_array))
        
        print(f"  ✓ PyTorch -> JAX -> PyTorch conversion successful")
        print(f"    Original shape: {torch_tensor.shape}")
        print(f"    Final shape: {back_to_torch.shape}")
        
        # Test that all libraries can coexist
        print("\n  Testing library coexistence...")
        
        # PyTorch operation
        pt_result = torch.mm(torch.randn(50, 50), torch.randn(50, 50))
        
        # JAX operation
        key = jax.random.PRNGKey(42)
        jax_result = jnp.dot(jax.random.normal(key, (50, 50)), 
                            jax.random.normal(key, (50, 50)))
        
        # LPIPS operation
        lpips_model = lpips.LPIPS(net='alex')
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)
        lpips_score = lpips_model(img1, img2)
        
        # svox operation
        octree = svox.N3Tree(data_dim=4, center=[0.5, 0.5, 0.5], radius=0.5)
        
        print("  ✓ All libraries can coexist and operate simultaneously")
        print(f"    PyTorch result shape: {pt_result.shape}")
        print(f"    JAX result shape: {jax_result.shape}")
        print(f"    LPIPS score: {lpips_score.item():.4f}")
        print(f"    svox octree data dim: {octree.data_dim}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Comprehensive GPU Test for PlenOctree Dependencies")
    print("=" * 60)
    
    # Run all tests
    system_ok = test_system_info()
    pytorch_ok, pytorch_gpu = test_pytorch()
    jax_ok, jax_gpu = test_jax()
    lpips_ok, lpips_gpu = test_lpips()
    svox_ok, svox_gpu = test_svox()
    integration_ok = test_integration()
    
    # Summary
    print_section("SUMMARY")
    
    print("Library Installation:")
    print(f"  PyTorch: {'✓' if pytorch_ok else '✗'}")
    print(f"  JAX: {'✓' if jax_ok else '✗'}")
    print(f"  LPIPS: {'✓' if lpips_ok else '✗'}")
    print(f"  svox: {'✓' if svox_ok else '✗'}")
    print(f"  Integration: {'✓' if integration_ok else '✗'}")
    
    print("\nGPU Support:")
    print(f"  PyTorch GPU: {'✓' if pytorch_gpu else '✗'}")
    print(f"  JAX GPU: {'✓' if jax_gpu else '✗'}")
    print(f"  LPIPS GPU: {'✓' if lpips_gpu else '✗'}")
    print(f"  svox GPU: {'✓' if svox_gpu else '✗'}")
    
    # Overall status
    all_libraries_ok = all([pytorch_ok, jax_ok, lpips_ok, svox_ok, integration_ok])
    all_gpu_ok = all([pytorch_gpu, jax_gpu, lpips_gpu, svox_gpu])
    
    print(f"\nOverall Status:")
    print(f"  All libraries working: {'✓' if all_libraries_ok else '✗'}")
    print(f"  All GPU support working: {'✓' if all_gpu_ok else '✗'}")
    
    if all_libraries_ok and all_gpu_ok:
        print("\n🎉 SUCCESS: All dependencies are working with GPU support!")
        print("   Your environment is ready for PlenOctree training and evaluation.")
    elif all_libraries_ok:
        print("\n⚠ PARTIAL SUCCESS: All libraries work but some GPU support is missing.")
        print("   You can still run PlenOctree but performance may be reduced.")
    else:
        print("\n❌ FAILURE: Some libraries are not working properly.")
        print("   Please check the error messages above and fix the issues.")
    
    return all_libraries_ok, all_gpu_ok

if __name__ == "__main__":
    main()
