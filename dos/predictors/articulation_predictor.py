import torch.nn as nn
import torch
import random
import numpy as np
import math

class ArticulationPredictor(nn.Module):
    
    def __init__(self, size_dataset, num_bones):
        super(ArticulationPredictor, self).__init__()
        
        degree = 10 #(0.17 rad)
        rad = degree * (math.pi/180)
        
        # Each bone_rotation is a 3D vector, 3 corresponds to x, y z coordinates
        # Using nn.Embedding to act as a lookup table
        self.bones_rotations = nn.Embedding(size_dataset, num_bones * 3) # Shape is torch.Size([49, 60]) 
        
        # Initialize the embedding weights to zeros
        # self.bones_rotations.weight.data.zero_()
        
        # Initialising the bone rotation parameter with zeros to make sure it starts from rest pose.
        # nn.init.zeros_(self.bones_rotations.weight)
        
        # nn.init.uniform_(self.bones_rotations.weight, -rad, rad)
        
        # Initializing the name to index dictionary
        self.name_to_index = {}
        
            
    def get_sample_index(self, names, device):
        # Sort names for consistent ordering
        names = sorted(names)
        
        indices = []
        for name in names:
            # If the name is not in the dictionary, assign it the next available index
            if name not in self.name_to_index:
                self.name_to_index[name] = len(self.name_to_index)
            indices.append(self.name_to_index[name])

        return torch.tensor(indices, device=device)


    def forward(self, batch, num_bones):
        # Extracting index of the samples from the batch using the sample names
        
        # degrees to radians
        degree = 10
        rad = degree * (math.pi/180)
        
        sample_names = batch['name']
        
        sample_index = self.get_sample_index(sample_names, batch['image'].device)

        # The embedding layer processes the batch of indices and retrieve the corresponding 
        # bones_rotations values for each sample in the batch
        bones_rotations = self.bones_rotations(sample_index)
        bones_rotations = bones_rotations.view(-1, num_bones, 3)
        
        # Applying tanh to ensure the values are between -1 and 1
        bones_rotations = torch.tanh(bones_rotations) * rad

        return bones_rotations
