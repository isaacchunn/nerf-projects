import torch
import torch.nn as nn

# Define a class for positional encoding, this is a technique to encode input data into a higher-dimensional space.
# This class builds a positional-encoding function that maps an input vector x in R^d to a higher dimensional space by concatenating
# 1. The raw input and
# 2. Multiple periodic transforms (e.g sin cos) of x at different frequencies.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
    
    def create_embedding_fn(self):
        # This method creates a list of functions that will transform input coordinates
        # In NeRF, we need to transform 3D coordinates (x,y,z) into higher dimensional space
        embed_fns = []  # Start with an empty list to store our transformation functions
        
        # Get the input dimensions from our configuration
        # For 3D coordinates, this would be d = 3 (x, y, z)
        d = self.kwargs['input_dims']
        
        # Keep track of how many dimensions our output will have
        # We start at 0 and will add dimensions as we add transformation functions
        out_dim = 0
        
        # Check if we should include the original input coordinates
        # This is important because sometimes we want both the original values AND the transformed ones
        if self.kwargs['include_input']:
            # Add a function that does nothing (identity function) - just returns the input as-is
            # This preserves the original coordinate information
            embed_fns.append(lambda x: x)
            # Increase our output dimension count by the input dimension
            # If input is 3D, this adds 3 more dimensions to our output
            out_dim += d

        # Get the maximum frequency from our configuration
        # This controls how "wiggly" our positional encoding will be
        # Higher max_freq means more detailed positional information
        max_freq = self.kwargs['max_freq_log2']
        
        # Get how many different frequencies we want to use
        # More frequencies = more detailed positional encoding
        N_freqs = self.kwargs['num_freqs']

        # Check if we want to sample frequencies logarithmically or linearly
        if self.kwargs['log_sampling']:
            # Logarithmic sampling. Convert to Python floats to avoid device mismatches later.
            freq_bands = (2. ** torch.linspace(0., max_freq, steps=N_freqs)).tolist()
        else:
            # Linear sampling. Convert to Python floats to avoid device mismatches later.
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs).tolist()
        
        # For each frequency we calculated above
        for freq in freq_bands:
            # For each periodic function (usually sin and cos) in our configuration
            for p_fn in self.kwargs['periodic_fns']:
                # Create a new function that applies the periodic function to input * frequency
                # This is the key insight: we multiply input by frequency before applying sin/cos
                # The lambda captures the current values of p_fn and freq for this iteration
                # freq is a Python float here, so x * freq keeps device of x
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                # Each periodic function applied to each frequency adds 'd' dimensions to output
                # So if we have 3D input, 2 periodic functions (sin, cos), and 10 frequencies:
                # We get 3 * 2 * 10 = 60 additional dimensions!
                out_dim += d
        
        # Store our list of transformation functions for later use
        self.embed_fns = embed_fns
        # Store the total output dimension so other parts of the code know what to expect
        self.out_dim = out_dim
    
    def embed(self, inputs):
        # This method actually applies all our transformation functions to the input
        # inputs: usually a batch of 3D coordinates (x, y, z)
        
        # For each transformation function in our list
        # Apply it to the inputs and collect all the results
        # torch.cat concatenates all results along the last dimension (-1)
        # This gives us one big vector with all the transformed coordinates
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    # This is a factory function that creates and configures positional encoders
    # multires: controls how many frequency bands we use (higher = more detailed encoding)
    # i: index parameter (usually 0, but -1 means "no encoding")
    
    # Special case: if i == -1, return no positional encoding
    # This is useful when you want to test NeRF without positional encoding
    if i == -1:
        # nn.Identity() does nothing - just returns input as-is
        # Return 3 because we're still working with 3D coordinates (x, y, z)
        return nn.Identity(), 3

    # Create a dictionary of configuration parameters for our positional encoder
    embed_kwargs = {
        'include_input': True,        # Keep the original 3D coordinates
        'input_dims': 3,              # We're encoding 3D coordinates (x, y, z)
        'max_freq_log2': multires-1,  # Maximum frequency = 2^(multires-1)
        'num_freqs': multires,        # How many different frequencies to use
        'log_sampling': True,         # Use logarithmic frequency spacing (better for NeRF)
        'periodic_fns': [torch.sin, torch.cos],  # Use both sine and cosine functions
    }
    
    # Create an Embedder object with our configuration
    # **embed_kwargs unpacks the dictionary into keyword arguments
    embedder_obj = Embedder(**embed_kwargs)
    
    # Create a convenient lambda function that applies the embedding
    # This lambda captures the embedder_obj so we can use it later
    # The lambda takes input x and applies the embedding5 function
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    
    # Return both the embedding function and the output dimension
    # The output dimension tells other parts of NeRF how big the encoded vectors will be
    # For example: if multires=10, output_dim = 3 + 2*3*10 = 63 dimensions
    return embed, embedder_obj.out_dim
