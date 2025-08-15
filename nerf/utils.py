"""
This file contains utility functions for the NeRF project.
"""
import yaml
import os
from typing import Dict, Any, Optional

def load_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load and parse YAML file
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        if config is None:
            raise yaml.YAMLError("Configuration file is empty")
            
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading config {config_path}: {e}")

def save_yaml(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save
        config_path (str): Path where to save the YAML file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save configuration to YAML file
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
            
    except Exception as e:
        raise Exception(f"Error saving config to {config_path}: {e}")

def create_default_config() -> Dict[str, Any]:
    """
    Create a default NeRF configuration. These are what i Found to be the default values for NeRF
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        # Basic experiment settings
        'expname': 'nerf_experiment',      # Experiment name
        'basedir': './logs/',              # Where to store checkpoints and logs
        'datadir': './data/llff/fern',     # Input data directory
        
        # Model architecture
        'netdepth': 8,                     # Layers in network (coarse)
        'netwidth': 256,                   # Channels per layer (coarse)
        'netdepth_fine': 8,                # Layers in fine network
        'netwidth_fine': 256,              # Channels per layer in fine network
        
        # Training options
        'N_rand': 32*32*4,                # Batch size (random rays per gradient step)
        'lrate': 5e-4,                    # Learning rate
        'lrate_decay': 250,               # Exponential learning rate decay (in 1000 steps)
        'chunk': 1024*32,                 # Rays processed in parallel (decrease if OOM)
        'netchunk': 1024*64,              # Points sent through network in parallel
        'no_batching': False,              # Only take random rays from 1 image at a time
        'no_reload': False,                # Do not reload weights from saved checkpoint
        'ft_path': None,                   # Specific weights file to reload for coarse network
        
        # Rendering options
        'N_samples': 64,                   # Number of coarse samples per ray
        'N_importance': 0,                 # Number of additional fine samples per ray
        'perturb': 1.0,                    # Set to 0 for no jitter, 1 for jitter
        'use_viewdirs': False,             # Use full 5D input instead of 3D
        'i_embed': 0,                      # Set 0 for default positional encoding, -1 for none
        'multires': 10,                    # Log2 of max freq for positional encoding (3D location)
        'multires_views': 4,               # Log2 of max freq for positional encoding (2D direction)
        'raw_noise_std': 0.0,              # Std dev of noise added to regularize sigma_a output
        
        # Rendering modes
        'render_only': False,              # Do not optimize, reload weights and render
        'render_test': False,              # Render the test set instead of render_poses path
        'render_factor': 0,                # Downsampling factor to speed up rendering
        
        # Training options (advanced)
        'precrop_iters': 0,                # Number of steps to train on central crops
        'precrop_frac': 0.5,               # Fraction of image taken for central crops
        
        # Dataset options
        'dataset_type': 'llff',            # Options: llff / blender / deepvoxels
        'testskip': 8,                     # Will load 1/N images from test/val sets
        
        # DeepVoxels flags
        'shape': 'greek',                  # Options: armchair / cube / greek / vase
        
        # Blender flags
        'white_bkgd': False,               # Set to render synthetic data on white background
        'half_res': False,                 # Load blender data at 400x400 instead of 800x800
        
        # LLFF flags
        'factor': 8,                       # Downsample factor for LLFF images
        'no_ndc': False,                   # Do not use normalized device coordinates
        'lindisp': False,                  # Sample linearly in disparity rather than depth
        'spherify': False,                 # Set for spherical 360 scenes
        'llffhold': 8,                     # Will take every 1/N images as LLFF test set
        
        # Logging and saving options
        'i_print': 100,                    # Frequency of console printout and metric logging
        'i_img': 500,                      # Frequency of tensorboard image logging
        'i_weights': 10000,                # Frequency of weight checkpoint saving
        'i_testset': 50000,                # Frequency of testset saving
        'i_video': 50000                   # Frequency of render_poses video saving
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that a configuration dictionary has all required keys.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # List of required configuration keys
    required_keys = [
        'expname', 'basedir', 'datadir',
        'netdepth', 'netwidth', 'netdepth_fine', 'netwidth_fine',
        'N_rand', 'lrate', 'lrate_decay', 'chunk', 'netchunk',
        'N_samples', 'N_importance', 'perturb', 'use_viewdirs',
        'i_embed', 'multires', 'multires_views', 'raw_noise_std',
        'dataset_type'
    ]
    
    # Check for missing required keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"Missing required configuration keys: {missing_keys}")
        return False
    
    # Validate specific parameter values
    if config['netdepth'] <= 0:
        print("Network depth (netdepth) must be positive")
        return False
        
    if config['netwidth'] <= 0:
        print("Network width (netwidth) must be positive")
        return False
        
    if config['lrate'] <= 0:
        print("Learning rate (lrate) must be positive")
        return False
        
    if config['N_samples'] <= 0:
        print("Number of samples (N_samples) must be positive")
        return False
        
    if config['chunk'] <= 0:
        print("Chunk size (chunk) must be positive")
        return False
        
    return True

def load_or_create_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file or create default if file doesn't exist.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        config = load_yaml(config_path)
    else:
        print(f"Configuration file {config_path} not found, creating default config")
        config = create_default_config()
        save_yaml(config, config_path)
        print(f"Default configuration saved to {config_path}")
    
    # Validate the configuration
    if not validate_config(config):
        print("Warning: Configuration validation failed")
    else:
        print("Configuration validation passed! Arguments are valid and correctly set.")
    
    return config
    
