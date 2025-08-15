import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# The NeRF model - This is the core neural network that learns to represent 3D scenes
# NeRF stands for "Neural Radiance Fields" - it learns to predict color and density at every 3D point
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        Initialize the NeRF neural network
        D: Number of layers in the main network (default: 8)
        W: Width of each layer (number of neurons, default: 256)
        input_ch: Input channels for 3D coordinates (default: 3 for x,y,z)
        input_ch_views: Input channels for viewing direction (default: 3 for dx,dy,dz)
        output_ch: Output channels (default: 4 for RGB + alpha)
        skips: Which layers to add skip connections (default: layer 4)
        use_viewdirs: Whether to use viewing direction (default: False)
        """
        super(NeRF, self).__init__()  # Initialize the parent PyTorch Module class
        
        # Store all the configuration parameters as instance variables
        self.D = D                    # Number of layers
        self.W = W                    # Width of each layer
        self.input_ch = input_ch      # How many dimensions for 3D coordinates
        self.input_ch_views = input_ch_views  # How many dimensions for viewing direction
        self.skips = skips            # Which layers get skip connections
        self.use_viewdirs = use_viewdirs      # Whether to use viewing direction
        
        # Create the main network layers that process 3D coordinates
        # This is the "coordinate network" that learns the 3D scene structure
        self.pts_linears = nn.ModuleList(
            # First layer: transforms input coordinates to hidden dimension W
            [nn.Linear(input_ch, W)] + 
            # Remaining layers: either normal layer or layer with skip connection
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # Create the viewing direction network (only used if use_viewdirs=True)
        # This network processes the viewing direction to determine color
        # According to the official NeRF implementation
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        # Alternative implementation from the paper (commented out)
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        # Create output layers based on whether we're using viewing directions
        if use_viewdirs:
            # If using viewing directions, we need separate outputs for alpha and RGB
            self.feature_linear = nn.Linear(W, W)      # Extract features from coordinate network
            self.alpha_linear = nn.Linear(W, 1)        # Predict alpha (transparency/density)
            self.rgb_linear = nn.Linear(W//2, 3)       # Predict RGB color (3 channels)
        else:
            # If not using viewing directions, just one output layer
            self.output_linear = nn.Linear(W, output_ch)  # Predict all outputs at once

    def forward(self, x):
        # This is the main forward pass - how data flows through the network
        # x contains both 3D coordinates and viewing direction concatenated together
        
        # Split the input into 3D coordinates and viewing direction
        # input_pts: 3D coordinates (x,y,z) that get positional encoding
        # input_views: viewing direction (dx,dy,dz) - where the camera is looking
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # Start processing the 3D coordinates through the main network
        h = input_pts  # h is our "hidden state" that gets updated through each layer
        
        # Process through all the coordinate network layers
        for i, l in enumerate(self.pts_linears):
            # Apply the current layer
            h = self.pts_linears[i](h)
            # Apply ReLU activation function (makes negative values 0, keeps positive values)
            h = F.relu(h)
            
            # If this is a skip connection layer, concatenate the original input
            # Skip connections help with gradient flow and learning deep networks
            # They're like "shortcuts" that let information flow directly from input to later layers
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)  # Concatenate along the last dimension

        # Now handle the viewing direction if we're using it
        if self.use_viewdirs:
            # Extract alpha (transparency/density) from the coordinate network output
            # Alpha tells us whether this 3D point is solid or empty
            alpha = self.alpha_linear(h)
            
            # Extract features that will be combined with viewing direction
            feature = self.feature_linear(h)
            
            # Combine the features with the viewing direction
            # This lets the network learn view-dependent effects (like reflections, highlights)
            h = torch.cat([feature, input_views], -1)
        
            # Process the combined features + viewing direction through the view network
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            # Finally, predict the RGB color based on viewing direction
            rgb = self.rgb_linear(h)
            
            # Combine RGB and alpha into the final output
            # RGB: what color this 3D point appears from this viewing angle
            # Alpha: how solid/transparent this 3D point is
            outputs = torch.cat([rgb, alpha], -1)
        else:
            # If not using viewing directions, just output everything from the coordinate network
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        # This method loads pre-trained weights from a Keras model
        # It's useful if you have a NeRF model trained in TensorFlow/Keras and want to use it in PyTorch
        
        # This only works when using viewing directions
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load weights for the coordinate network layers (pts_linears)
        for i in range(self.D):
            idx_pts_linears = 2 * i  # Keras stores weights and biases separately
            # Load the weight matrix (transpose because Keras and PyTorch use different conventions)
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            # Load the bias vector
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load weights for the feature extraction layer
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load weights for the viewing direction network
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load weights for the RGB prediction layer
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load weights for the alpha prediction layer
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
