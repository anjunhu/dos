import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
import faiss
import cv2
import os
from matplotlib.patches import ConnectionPatch
import io

#---- This func is written by Oishi
# Function to pad a tensor with zeros to the max_length
def padding_tensor(tensor, max_length, device):
    n = tensor.shape[0]
    if n < max_length:
        padding = max_length - n
        # Ensuring the padding tensor is on the same device as the input tensor
        padding_tensor = torch.zeros(padding, 2, device=device)
        padded_tensor = torch.cat((tensor, padding_tensor), dim=0)
        return padded_tensor
    return tensor

#---- This func is written by Oishi
def tensor_to_matplotlib_figure(tensor):
    # Convert PyTorch tensor to numpy array
    # If the tensor is on GPU, move it to CPU and then convert to numpy
    numpy_image = tensor.detach().cpu().numpy()

    # Convert from CHW to HWC format for matplotlib
    numpy_image = np.transpose(numpy_image, (1, 2, 0))

    # Create a matplotlib figure
    fig = plt.figure()
    plt.imshow(numpy_image)
    plt.axis('off')  # Turn off axis

    return fig

#---- This func is written by Oishi
def draw_correspondences_1_image(points1: List[Tuple[float, float]], image1: Image.Image, index=0) -> plt.Figure:

    num_points = len(points1)

    if num_points > 15:
        cmap = plt.get_cmap('viridis')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "white", "black", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    colors = cmap(np.linspace(0, 1, num_points))
    radius1, radius2 = 0.03*max(image1.size), 0.01*max(image1.size)
    
    # plot a subfigure put image1 in the top, image2 in the bottom
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.axis('off')

    for idx, (point1, color) in enumerate(zip(points1, colors)):
        x1, y1  = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        #ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        
        # Adding an integer number next to the point
        ax1.text(x1 + radius2, y1, str(idx), color='blue', fontsize=10, verticalalignment='center', horizontalalignment='left')
    
    # ax1.text(0.7, 0.95, f' Frame_{index}', verticalalignment='top', horizontalalignment='center', color = 'black', fontsize ='13', fontweight = 'bold', transform=ax1.transAxes)
    ax1.imshow(image1)
    plt.tight_layout()
    
    return fig

def convert_fig_to_image(fig):
    """
    Convert a Matplotlib figure to a PIL Image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Close the figure to prevent it from displaying in a notebook or GUI
    return img


def draw_correspondences_combined(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]], 
                                  image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Draw point correspondences on images and return a combined image.
    """
    # Set the gap width
    small_gap_width = 40
    
    # Adjust the figure size based on the input image height
    fig, axs = plt.subplots(1, 2, figsize=(12, image1.height / 100))
    
    for ax, points, image in zip(axs, [points1, points2], [image1, image2]):
        ax.imshow(image)
        ax.axis('off')
        num_points = len(points)
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                               "maroon", "white", "black", "chocolate", "gray", "blueviolet"])
        colors = cmap(np.linspace(0, 1, num_points))
        radius = 0.01 * max(image.size)
        
        for idx, (point, color) in enumerate(zip(points, colors)):
            x, y = point
            circ = plt.Circle((x, y), radius, color=color)
            ax.add_patch(circ)
            
            # Adding an integer number next to the point
            ax.text(x + radius, y, str(idx), color='blue', fontsize=10, verticalalignment='center', horizontalalignment='left')
    
    # Convert the figure with plots to a PIL image
    fig1_image = convert_fig_to_image(fig)

    # Resize fig1_image to match the height of the input images
    fig1_image = fig1_image.resize((image1.width * 2, image1.height), Image.Resampling.LANCZOS)
    
    # Calculate total width considering
    total_width = image1.width * 2 + image2.width * 2   # + small_gap_width 

    # Create a new image with a width that can hold both the input images and the plot images.
    combined_image = Image.new('RGB', (total_width, image1.height), 'black')
    
    # Paste the images into the combined image with specified gaps
    combined_image.paste(image1, (0, 0))
    combined_image.paste(Image.new('RGB', (small_gap_width, image1.height), 'white'), (image1.width, 0)) 
    combined_image.paste(fig1_image.crop((0, 0, image1.width, image1.height)), (image1.width + small_gap_width, 0))
    # combined_image.paste(Image.new('RGB', (small_gap_width, image1.height), 'white'), (image1.width * 2, 0))
    combined_image.paste(image2, (image1.width * 2, 0))
    # combined_image.paste(Image.new('RGB', (small_gap_width, image1.height), 'white'), (image1.width * 3, 0))
    combined_image.paste(fig1_image.crop((image1.width, 0, image1.width * 2, image1.height)), (image1.width * 3, 0))
    
    return combined_image


##----- This func is taken from the "Tale of 2 Features" paper (https://github.com/Junyi42/sd-dino/blob/master/utils/utils_correspondence.py)
def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS) # Image.Resampling.LANCZOS
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS) # Image.Resampling.LANCZOS
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS) # Image.Resampling.LANCZOS
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS) # Image.Resampling.LANCZOS
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

##----- This func is taken from the "Tale of 2 Features" paper (https://github.com/Junyi42/sd-dino/blob/master/utils/utils_correspondence.py)
def find_nearest_patchs(mask1, mask2, image1, image2, features1, features2, mask=False, resolution=None, edit_image=None):
    
    def polar_color_map(image_shape):
        h, w = image_shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Find the center of the mask
        mask=mask2.cpu()
        mask_center = np.array(np.where(mask > 0))
        mask_center = np.round(np.mean(mask_center, axis=1)).astype(int)
        mask_center_y, mask_center_x = mask_center

        # Calculate distance and angle based on mask_center
        xx_shifted, yy_shifted = xx - x[mask_center_x], yy - y[mask_center_y]
        max_radius = np.sqrt(h**2 + w**2) / 2
        radius = np.sqrt(xx_shifted**2 + yy_shifted**2) * max_radius
        angle = np.arctan2(yy_shifted, xx_shifted) / (2 * np.pi) + 0.5

        angle = 0.2 + angle * 0.6  # Map angle to the range [0.25, 0.75]
        radius = np.where(radius <= max_radius, radius, max_radius)  # Limit radius values to the unit circle
        radius = 0.2 + radius * 0.6 / max_radius  # Map radius to the range [0.1, 1]

        return angle, radius
    
    if resolution is not None: # resize the feature map to the resolution
        features1 = F.interpolate(features1, size=resolution, mode='bilinear')
        features2 = F.interpolate(features2, size=resolution, mode='bilinear')
    
    # resize the image to the shape of the feature map
    resized_image1 = resize(image1, features1.shape[2], resize=True, to_pil=False)
    resized_image2 = resize(image2, features2.shape[2], resize=True, to_pil=False)

    if mask: # mask the features
        resized_mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=features1.shape[2:], mode='nearest')
        resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=features2.shape[2:], mode='nearest')
        features1 = features1 * resized_mask1.repeat(1, features1.shape[1], 1, 1)
        features2 = features2 * resized_mask2.repeat(1, features2.shape[1], 1, 1)
        # set where mask==0 a very large number
        features1[(features1.sum(1)==0).repeat(1, features1.shape[1], 1, 1)] = 100000
        features2[(features2.sum(1)==0).repeat(1, features2.shape[1], 1, 1)] = 100000

    features1_2d = features1.reshape(features1.shape[1], -1).permute(1, 0).cpu().detach().numpy()
    features2_2d = features2.reshape(features2.shape[1], -1).permute(1, 0).cpu().detach().numpy()

    features1_2d = torch.tensor(features1_2d).to("cuda")
    features2_2d = torch.tensor(features2_2d).to("cuda")
    resized_image1 = torch.tensor(resized_image1).to("cuda").float()
    resized_image2 = torch.tensor(resized_image2).to("cuda").float()

    mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image1.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image2.shape[:2], mode='nearest').squeeze(0).squeeze(0)

    # Mask the images
    resized_image1 = resized_image1 * mask1.unsqueeze(-1).repeat(1, 1, 3)
    resized_image2 = resized_image2 * mask2.unsqueeze(-1).repeat(1, 1, 3)
    # Normalize the images to the range [0, 1]
    resized_image1 = (resized_image1 - resized_image1.min()) / (resized_image1.max() - resized_image1.min())
    resized_image2 = (resized_image2 - resized_image2.min()) / (resized_image2.max() - resized_image2.min())

    angle, radius = polar_color_map(resized_image2.shape)

    angle_mask = angle * mask2.cpu().numpy()
    radius_mask = radius * mask2.cpu().numpy()

    hsv_mask = np.zeros(resized_image2.shape, dtype=np.float32)
    hsv_mask[:, :, 0] = angle_mask
    hsv_mask[:, :, 1] = radius_mask
    hsv_mask[:, :, 2] = 1

    rainbow_mask2 = cv2.cvtColor((hsv_mask * 255).astype(np.uint8), cv2.COLOR_HSV2BGR) / 255

    if edit_image is not None:
        rainbow_mask2 = cv2.imread(edit_image, cv2.IMREAD_COLOR)
        rainbow_mask2 = cv2.cvtColor(rainbow_mask2, cv2.COLOR_BGR2RGB) / 255
        rainbow_mask2 = cv2.resize(rainbow_mask2, (resized_image2.shape[1], resized_image2.shape[0]))

    # Apply the rainbow mask to image2
    rainbow_image2 = rainbow_mask2 * mask2.cpu().numpy()[:, :, None]

    # Create a white background image
    background_color = np.array([1, 1, 1], dtype=np.float32)
    background_image = np.ones(resized_image2.shape, dtype=np.float32) * background_color

    # Apply the rainbow mask to image2 only in the regions where mask2 is 1
    rainbow_image2 = np.where(mask2.cpu().numpy()[:, :, None] == 1, rainbow_mask2, background_image)
    
    nearest_patches = []

    distances = torch.cdist(features1_2d, features2_2d)
    nearest_patch_indices = torch.argmin(distances, dim=1)
    nearest_patches = torch.index_select(torch.tensor(rainbow_mask2).cuda().reshape(-1, 3), 0, nearest_patch_indices)

    nearest_patches_image = nearest_patches.reshape(resized_image1.shape)
    rainbow_image2 = torch.tensor(rainbow_image2).to("cuda")

    # TODO: upsample the nearest_patches_image to the resolution of the original image
    # nearest_patches_image = F.interpolate(nearest_patches_image.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)
    # rainbow_image2 = F.interpolate(rainbow_image2.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)

    nearest_patches_image = (nearest_patches_image).cpu().numpy()
    resized_image2 = (rainbow_image2).cpu().numpy()

    return nearest_patches_image, resized_image2, nearest_patches

##----- This func is taken from the "Tale of 2 Features" paper (https://github.com/Junyi42/sd-dino/blob/master/utils/utils_correspondence.py)
def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


###----- This func is taken from the "Tale of 2 Features" paper (https://github.com/Junyi42/sd-dino/blob/master/utils/utils_correspondence.py)

# The function takes two input tensors x and y, which are expected to be 3-dimensional tensors. 
# The parameter p specifies the power for the Minkowski distance (defaults to 2, which is the Euclidean distance), 
# and normalize is a boolean flag indicating whether to normalize the feature vectors before computing the distances. 
# The function returns a tensor representing the pairwise similarities between x and y.

def pairwise_sim(x: torch.Tensor, y: torch.Tensor, p=2, normalize=False) -> torch.Tensor:
    # compute similarity based on euclidean distances
    
    # x and y are 3-dimensional tensors.
    # Normalize the feature vectors in x and y respectively along the last dimension (i.e., the feature dimension) 
    # using the L2 norm. This step ensures that each feature vector has a unit L2 norm.
    if normalize:
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
        
    # result_list is of 3 dimension - [batch_size, sequence_length, num_token_x]    
    result_list=[]
    
    # Get the number of tokens (or patches) in the x tensor. The assumption is that the tensors x and y have the same number of tokens.
    num_token_x = x.shape[2]
    
    # Loop over each token (patch) in the x tensor.
    for token_idx in range(num_token_x):
        """Extract the feature vector of the current token from the x tensor and add a new dimension to it using unsqueeze. 
        The resulting tensor token has shape [batch_size, sequence_length, 1, feature_dim].
        The dimension of the 'token' is 4."""
        token = x[:, :, token_idx, :].unsqueeze(dim=2)
        """ Compute the pairwise distances between the feature vector token and the feature vectors in y. The torch.nn.PairwiseDistance function computes the 
        Minkowski distance between the two tensors using the specified p value (in this case, the Euclidean distance). The distances are negated (*(-1)) to 
        obtain similarities instead of distances, and the resulting similarity tensor is appended to result_list."""
        result_list.append(torch.nn.PairwiseDistance(p=p)(token, y)*(-1))
        
    #After the loop, result_list contains a list of tensors, where each tensor represents the similarities between one token from x and all the tokens in y.
    """Stack the tensors in result_list along the third dimension to create the final output tensor, where each element at position [batch_size, sequence_length, i] 
    represents the similarity between the i-th token in x and all tokens in y. The output tensor has shape [batch_size, sequence_length, num_token_x]."""
    return torch.stack(result_list, dim=2)


##----- This func is taken from the "Tale of 2 Features" paper (https://github.com/Junyi42/sd-dino/blob/master/utils/utils_correspondence.py)
def co_pca(features1, features2, dim=[128,128,128]):
    
    processed_features1 = {}
    processed_features2 = {}
    s5_size = features1['s5'].shape[-1]
    s4_size = features1['s4'].shape[-1]
    s3_size = features1['s3'].shape[-1]
    # Get the feature tensors
    s5_1 = features1['s5'].reshape(features1['s5'].shape[0], features1['s5'].shape[1], -1)
    s4_1 = features1['s4'].reshape(features1['s4'].shape[0], features1['s4'].shape[1], -1)
    s3_1 = features1['s3'].reshape(features1['s3'].shape[0], features1['s3'].shape[1], -1)

    s5_2 = features2['s5'].reshape(features2['s5'].shape[0], features2['s5'].shape[1], -1)
    s4_2 = features2['s4'].reshape(features2['s4'].shape[0], features2['s4'].shape[1], -1)
    s3_2 = features2['s3'].reshape(features2['s3'].shape[0], features2['s3'].shape[1], -1)
    # Define the target dimensions
    target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2]}

    # Compute the PCA
    for name, tensors in zip(['s5', 's4', 's3'], [[s5_1, s5_2], [s4_1, s4_2], [s3_1, s3_2]]):
        target_dim = target_dims[name]

        # Concatenate the features
        features = torch.cat(tensors, dim=-1) # along the spatial dimension
        features = features.permute(0, 2, 1) # Bx(t_x+t_y)x(d)

        # Compute the PCA
        # pca = faiss.PCAMatrix(features.shape[-1], target_dim)

        # Train the PCA
        # pca.train(features[0].cpu().numpy())

        # Apply the PCA
        # features = pca.apply(features[0].cpu().numpy()) # (t_x+t_y)x(d)

        # convert to tensor
        # features = torch.tensor(features, device=features1['s5'].device).unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        
        
        # equivalent to the above, pytorch implementation
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered_features = features[0] - mean
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)
        features = reduced_features.unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        

        # Split the features
        processed_features1[name] = features[:, :, :features.shape[-1] // 2] # Bx(d)x(t_x)
        processed_features2[name] = features[:, :, features.shape[-1] // 2:] # Bx(d)x(t_y)

    # reshape the features
    processed_features1['s5']=processed_features1['s5'].reshape(processed_features1['s5'].shape[0], -1, s5_size, s5_size)
    processed_features1['s4']=processed_features1['s4'].reshape(processed_features1['s4'].shape[0], -1, s4_size, s4_size)
    processed_features1['s3']=processed_features1['s3'].reshape(processed_features1['s3'].shape[0], -1, s3_size, s3_size)

    processed_features2['s5']=processed_features2['s5'].reshape(processed_features2['s5'].shape[0], -1, s5_size, s5_size)
    processed_features2['s4']=processed_features2['s4'].reshape(processed_features2['s4'].shape[0], -1, s4_size, s4_size)
    processed_features2['s3']=processed_features2['s3'].reshape(processed_features2['s3'].shape[0], -1, s3_size, s3_size)

    # Upsample s5 spatially by a factor of 2
    processed_features1['s5'] = F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    processed_features2['s5'] = F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features1['s5'] = torch.cat([processed_features1['s4'], processed_features1['s5']], dim=1)
    processed_features2['s5'] = torch.cat([processed_features2['s4'], processed_features2['s5']], dim=1)

    # Set s3 as the new s4
    processed_features1['s4'] = processed_features1['s3']
    processed_features2['s4'] = processed_features2['s3']

    # Remove s3 from the features dictionary
    processed_features1.pop('s3')
    processed_features2.pop('s3')

    # current order are layer 8, 5, 2
    features1_gether_s4_s5 = torch.cat([processed_features1['s4'], F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear')], dim=1)
    features2_gether_s4_s5 = torch.cat([processed_features2['s4'], F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear')], dim=1)

    return features1_gether_s4_s5, features2_gether_s4_s5
