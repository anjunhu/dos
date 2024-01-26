##------- CODE mainly taken from the "Tale of 2 Features" paper (https://github.com/Junyi42/sd-dino/blob/master/pck_spair_pascal.py), 
##------- but further modifications have been made to adapt to our project.----------------------------------------------------------

import sys
import os
import torch
from dos.components.fuse.extractor_dino import ViTExtractor
from dos.components.fuse.extractor_sd import process_features_and_mask, get_mask
from tqdm import tqdm
from dos.utils.correspondence import pairwise_sim, chunk_cosine_sim, co_pca, find_nearest_patchs, \
                                resize, draw_correspondences_1_image  #, get_n_colors
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import ipdb
import time


# NOT_FUSE = False
# ONLY_DINO = False
# DINOV1 = False
# FUSE_DINO = False if NOT_FUSE else True
# DINOV2 = False if DINOV1 else True
# CO_PCA_DINO = 0
# CO_PCA = True
# MODEL_SIZE = 'base'
# TEXT_INPUT = False
# EDGE_PAD = False
# # set true to use the raw features from sd
# RAW = False
# # the dimensions of the three groups of sd features
# PCA_DIMS =[256, 256, 256]
# # first three correspond to three layers for the sd features, and the last two for the ensembled sd/dino features
# WEIGHT =[1,1,1,1,1]
# MASK = False

# # SIZE=960; # image size for the sd input # ORIGINAL CODE
# model_dict={'small':'dinov2_vits14',
#                 'base':'dinov2_vitb14',
#                 'large':'dinov2_vitl14',
#                 'giant':'dinov2_vitg14'}

# model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits8'
# stride = 14 if DINOV2 else 4 if ONLY_DINO else 8
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# start_time = time.time()
# extractor = ViTExtractor(model_type, stride, device=device)
# end_time = time.time()
# # # Open a file in append mode
# # with open('log.txt', 'a') as file:
# #     file.write(f"The ViTExtractor function took {end_time - start_time} seconds to run.\n")
# print(f'The ViTExtractor function took {end_time - start_time} seconds to run.')

class ComputeCorrespond:
        
    def __init__(self):
        self.ONLY_DINO = False
        self.DINOV1 = False
        self.DINOV2 = False if self.DINOV1 else True
        self.MODEL_SIZE = 'base'
        self.model_dict = {'small': 'dinov2_vits14',
                  'base': 'dinov2_vitb14',
                  'large': 'dinov2_vitl14',
                  'giant': 'dinov2_vitg14'}
        self.model_type = self.model_dict[self.MODEL_SIZE] if self.DINOV2 else 'dino_vits8'
        self.stride = 14 if self.DINOV2 else 4 if self.ONLY_DINO else 8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.TEXT_INPUT = False
        self.EDGE_PAD = False
        self.NOT_FUSE = False
        self.FUSE_DINO = False if self.NOT_FUSE else True
        self.CO_PCA_DINO = 0
        self.CO_PCA = True
        self.RAW = False
        self.PCA_DIMS = [256, 256, 256]
        self.WEIGHT = [1, 1, 1, 1, 1]
        self.MASK = False
        
        self.start_time = time.time()
        self.extractor = ViTExtractor(self.model_type, self.stride, device=self.device)
        self.end_time = time.time()
        print(f'The ViTExtractor function took {self.end_time - self.start_time} seconds to run.')

      
    def compute_correspondences_sd_dino(self, img1, img1_kps, img2, index, model, aug, files=None, category='horse', mask=False, dist='l2', thresholds=None, real_size=960):  # kps,
        # print('compute_correspondences func is running...')
        
        img_size = 840 if self.DINOV2 else 240 if self.ONLY_DINO else 480    # ORIGINAL CODE # should it be 224 or 240, because 60 * stride(i.e 4) is 240
        
        layer = 11 if self.DINOV2 else 9
        if 'l' in self.model_type:
            layer = 23
        elif 'g' in self.model_type:
            layer = 39
        facet = 'token' if self.DINOV2 else 'key'
        
        patch_size = self.extractor.model.patch_embed.patch_size[0] if self.DINOV2 else self.extractor.model.patch_embed.patch_size
        num_patches = int(patch_size / self.stride * (img_size // patch_size - 1) + 1)   # num_patches is 60
        
        input_text = "a photo of "+category if self.TEXT_INPUT else None

        if thresholds is not None:
            thresholds = torch.tensor(thresholds).to(self.device)

        # Load image 1
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=self.EDGE_PAD) # this is for sd - img size used is 960*960
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=self.EDGE_PAD)        # this is for DINO - img size used is 840*840
        
        # Get patch index for the keypoints
        img1_y = img1_kps[:, 1].cpu()           # ORIGINAL CODE                # img1_kps should be [(num of kps)20,2]
        img1_x = img1_kps[:, 0].cpu()           # ORIGINAL CODE                # img1_kps should be [(num of kps)20,2]                    
        img1_y, img1_x = img1_y.detach().numpy(), img1_x.detach().numpy()
        img1_y_patch = (num_patches / img_size * img1_y).astype(np.int32)
        img1_x_patch = (num_patches / img_size * img1_x).astype(np.int32)
        img1_patch_idx = num_patches * img1_y_patch + img1_x_patch
        
        # Load image 2
        img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=self.EDGE_PAD)
        
        img2 = resize(img2, img_size, resize=True, to_pil=True, edge=self.EDGE_PAD)
        
        with torch.no_grad():
            if not self.CO_PCA:
                if not self.ONLY_DINO:
                    # print('Its not ONLY_DINO and not CO_PCA')
                    img1_desc = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                    img2_desc = process_features_and_mask(model, aug, img2_input, category, input_text=input_text,  mask=mask).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                if self.FUSE_DINO:
                    # print('Its using FUSE_DINO and not CO_PCA')
                    img1_batch = self.extractor.preprocess_pil(img1)
                    img1_desc_dino = self.extractor.extract_descriptors(img1_batch.to(self.device), layer, facet)
                    img2_batch = self.extractor.preprocess_pil(img2)
                    img2_desc_dino = self.extractor.extract_descriptors(img2_batch.to(self.device), layer, facet)
            else:
                if not self.ONLY_DINO:
                    ## This extracts the sd features
                    # print('Its not ONLY_DINO and CO_PCA') # ->  this is run when used FUSE = sd + DINOv2
                    start_time = time.time()
                    features1, features2 = process_features_and_mask(model, aug, img1_input, img2_input, input_text=input_text,  mask=False, raw=True)  # features1 is a dict
                    end_time = time.time()
                    # # Open a file in append mode
                    # with open('log.txt', 'a') as file:
                    #     file.write(f"The process_features_and_mask function for img 1 took {end_time - start_time} seconds to run.\n")
                    
                    print(f"The process_features_and_mask function for img 1 and 2 took {end_time - start_time} seconds to run.")
                    
                    if not self.RAW:
                        processed_features1, processed_features2 = co_pca(features1, features2, self.PCA_DIMS)  # processed_features1 shape is torch.Size([1, 768, 60, 60])
                    else:                                                                                  # processed_features2 shape is torch.Size([1, 768, 60, 60])
                        if self.WEIGHT[0]:
                            print('WEIGHT[0] is used')
                            processed_features1 = features1['s5']   # processed_features1 shape is torch.Size([1, 1280, 15, 15])
                            processed_features2 = features2['s5']   # processed_features2 shape is torch.Size([1, 1280, 15, 15])
                        elif self.WEIGHT[1]:
                            print('WEIGHT[1] is used')
                            processed_features1 = features1['s4']   # processed_features1 shape is torch.Size([1, 1280, 30, 30])
                            processed_features2 = features2['s4']   # processed_features1 shape is torch.Size([1, 1280, 30, 30])
                        elif self.WEIGHT[2]:
                            print('WEIGHT[2] is used')
                            processed_features1 = features1['s3']   # processed_features1 shape is torch.Size([1, 640, 60, 60])
                            processed_features2 = features2['s3']   # processed_features1 shape is torch.Size([1, 640, 60, 60])
                        elif self.WEIGHT[3]:
                            print('WEIGHT[3] is used')
                            processed_features1 = features1['s2']   # processed_features1 shape is torch.Size([1, 320, 120, 120])
                            processed_features2 = features2['s2']   # processed_features1 shape is torch.Size([1, 320, 120, 120])
                        else:
                            raise NotImplementedError
                        
                        # rescale the features
                        processed_features1 = F.interpolate(processed_features1, size=(num_patches, num_patches), mode='bilinear', align_corners=False) # torch.Size([1, 320, 60, 60])
                        processed_features2 = F.interpolate(processed_features2, size=(num_patches, num_patches), mode='bilinear', align_corners=False) # torch.Size([1, 640, 60, 60])
                    img1_desc = processed_features1.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)  # img1_desc shape torch.Size([1, 1, 3600, 320])
                    img2_desc = processed_features2.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)  # img2_desc shape torch.Size([1, 1, 3600, 640])
                
                if self.FUSE_DINO:
                    # This extracts the DINOv2 features
                    # print('Its using FUSE_DINO and CO_PCA') # ->  this is run when used FUSE = sd + DINOv2
                    start_time = time.time()
                    img1_batch = self.extractor.preprocess_pil(img1)    # time_taken: 0.0072109 sec              # img1_batch shape is torch.Size([1, 3, 840, 840])
                    end_time = time.time()
                    # # Open a file in append mode
                    # with open('log.txt', 'a') as file:
                    #     file.write(f"The DINOV2 extractor.preprocess_pil function for img1 took {end_time - start_time} seconds to run.\n")
                    
                    print(f"The DINOV2 extractor.preprocess_pil function for img1 took {end_time - start_time} seconds to run.")
                    
                    start_time = time.time()
                    img1_desc_dino = self.extractor.extract_descriptors(img1_batch.to(self.device), layer, facet)     # img1_desc_dino shape is torch.Size([1, 1, 3600, 768])
                    end_time = time.time()
                    # # Open a file in append mode
                    # with open('log.txt', 'a') as file:
                    #     file.write(f"The DINOV2 extractor.extract_descriptors function for img1 took {end_time - start_time} seconds to run.\n")
                    print(f"The DINOV2 extractor.extract_descriptors function for img1 took {end_time - start_time} seconds to run.")
                    
                    img2_batch = self.extractor.preprocess_pil(img2)                                             # img2_batch shape is torch.Size([1, 3, 840, 840])
                    
                    start_time = time.time()
                    img2_desc_dino = self.extractor.extract_descriptors(img2_batch.to(self.device), layer, facet)     # img2_desc_dino shape is torch.Size([1, 1, 3600, 768])
                    # # Open a file in append mode
                    # with open('log.txt', 'a') as file:
                    #     file.write(f"The DINOV2 extractor.extract_descriptors function for img2 took {end_time - start_time} seconds to run.\n")
                    print(f"The DINOV2 extractor.extract_descriptors function for img2 took {end_time - start_time} seconds to run.")
                    
            
            if self.CO_PCA_DINO:
                # print('Its using CO_PCA_DINO')
                cat_desc_dino = torch.cat((img1_desc_dino, img2_desc_dino), dim=2).squeeze() # (1, 1, num_patches**2, dim)
                mean = torch.mean(cat_desc_dino, dim=0, keepdim=True)
                centered_features = cat_desc_dino - mean
                U, S, V = torch.pca_lowrank(centered_features, q=self.CO_PCA_DINO)
                reduced_features = torch.matmul(centered_features, V[:, :self.CO_PCA_DINO]) # (t_x+t_y)x(d)
                processed_co_features = reduced_features.unsqueeze(0).unsqueeze(0)
                img1_desc_dino = processed_co_features[:, :, :img1_desc_dino.shape[2], :]
                img2_desc_dino = processed_co_features[:, :, img1_desc_dino.shape[2]:, :]
            if not self.ONLY_DINO and not self.RAW: # reweight different layers of sd
                # print('Its not ONLY_DINO and not RAW')                      # ->  this is run when used FUSE = sd + DINOv2
                
                start_time = time.time()
                img1_desc[...,:self.PCA_DIMS[0]]*=self.WEIGHT[0]                                                  # img1_desc shape torch.Size([1, 1, 3600, 320])
                img1_desc[...,self.PCA_DIMS[0]:self.PCA_DIMS[1]+self.PCA_DIMS[0]]*=self.WEIGHT[1]                           # img1_desc shape torch.Size([1, 1, 3600, 320])
                img1_desc[...,self.PCA_DIMS[1]+self.PCA_DIMS[0]:self.PCA_DIMS[2]+self.PCA_DIMS[1]+self.PCA_DIMS[0]]*=self.WEIGHT[2]   # img1_desc shape torch.Size([1, 1, 3600, 320])
                img2_desc[...,:self.PCA_DIMS[0]]*=self.WEIGHT[0]                                                  # img2_desc shape torch.Size([1, 1, 3600, 640])
                img2_desc[...,self.PCA_DIMS[0]:self.PCA_DIMS[1]+self.PCA_DIMS[0]]*=self.WEIGHT[1]                           # img2_desc shape torch.Size([1, 1, 3600, 640])
                img2_desc[...,self.PCA_DIMS[1]+self.PCA_DIMS[0]:self.PCA_DIMS[2]+self.PCA_DIMS[1]+self.PCA_DIMS[0]]*=self.WEIGHT[2]   # img2_desc shape torch.Size([1, 1, 3600, 640])
                
                end_time = time.time()
                # Open a file in append mode
                # with open('log.txt', 'a') as file:
                #     file.write(f"The 'reweight different layers of sd' took {end_time - start_time} seconds to run.\n")
                
            if 'l1' in dist or 'l2' in dist or dist == 'plus_norm':
                # print('Its uses dist as l1 or l2 or plus_norm')
                # normalize the features
                img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)                    # img1_desc shape is torch.Size([1, 1, 3600, 320])
                img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)                    # img2_desc shape is torch.Size([1, 1, 3600, 640])
                img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)     # img1_desc_dino shape is torch.Size([1, 1, 3600, 768])
                img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)     # img2_desc_dino shape is torch.Size([1, 1, 3600, 768])
            if self.FUSE_DINO and not self.ONLY_DINO and dist!='plus' and dist!='plus_norm':
                # print('Its FUSE_DINO and not ONLY_DINO and dist is not plus and not plus_norm') # ->  this is run when used FUSE = sd + DINOv2
                # concatenate two features i.e sd and dino features together
                img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)      # img1_desc torch.Size([1, 1, 3600, 1088])
                img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)      # img2_desc torch.Size([1, 1, 3600, 1408])
                if not self.RAW:
                    # print('Its FUSE_DINO and not ONLY_DINO and dist is not plus and not plus_norm and its not RAW') # ->  this is run when used FUSE = sd + DINOv2
                    start_time = time.time()
                    # reweight sd and dino
                    img1_desc[...,:self.PCA_DIMS[2]+self.PCA_DIMS[1]+self.PCA_DIMS[0]]*=self.WEIGHT[3]  # img1_desc torch.Size([1, 1, 3600, 1088])
                    img1_desc[...,self.PCA_DIMS[2]+self.PCA_DIMS[1]+self.PCA_DIMS[0]:]*=self.WEIGHT[4]  # img1_desc torch.Size([1, 1, 3600, 1088])
                    img2_desc[...,:self.PCA_DIMS[2]+self.PCA_DIMS[1]+self.PCA_DIMS[0]]*=self.WEIGHT[3]  # img2_desc torch.Size([1, 1, 3600, 1408])
                    img2_desc[...,self.PCA_DIMS[2]+self.PCA_DIMS[1]+self.PCA_DIMS[0]:]*=self.WEIGHT[4]  # img2_desc torch.Size([1, 1, 3600, 1408])
                    
                    end_time = time.time()
                    # # Open a file in append mode
                    # with open('log.txt', 'a') as file:
                    #     file.write(f"The 'reweight sd and dino' took {end_time - start_time} seconds to run.\n")
                        
            elif dist=='plus' or dist=='plus_norm':
                # print('Its FUSE_DINO and not ONLY_DINO and dist is plus or plus_norm')
                img1_desc = img1_desc + img1_desc_dino
                img2_desc = img2_desc + img2_desc_dino
                dist='cos'
            
            if self.ONLY_DINO:
                # print('Its ONLY_DINO')
                img1_desc = img1_desc_dino
                img2_desc = img2_desc_dino
            # logger.info(img1_desc.shape, img2_desc.shape)
            
        if self.MASK and self.CO_PCA:
            # print('It uses MASK and CO_PCA')
            mask2 = get_mask(model, aug, img2, category)
            img2_desc = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches)
            resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=(num_patches, num_patches), mode='nearest')
            img2_desc = img2_desc * resized_mask2.repeat(1, img2_desc.shape[1], 1, 1)
            img2_desc[(img2_desc.sum(dim=1)==0).repeat(1, img2_desc.shape[1], 1, 1)] = 100000
            # reshape back
            img2_desc = img2_desc.reshape(1, 1, img2_desc.shape[1], num_patches*num_patches).permute(0,1,3,2)

        # print('img1_desc shape', img1_desc.shape)  # img1_desc shape is torch.Size([1, 1, 3600, 1536]) the image dim is 60 * 60 = 3600, 60 is multiplied with stride 14 = 840, hence for DINOv2 the input_size is 840
        # print('img2_desc shape', img2_desc.shape)  # img2_desc shape is torch.Size([1, 1, 3600, 1536])
        
        # Get similarity matrix
        if dist == 'cos':
            # print('It uses cosine similarity')
            sim_1_to_2 = chunk_cosine_sim(img1_desc, img2_desc).squeeze()
        elif dist == 'l2':
            # print('It uses l2')
            start_time = time.time()
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=2).squeeze()
            end_time = time.time()
            # # Open a file in append mode
            # with open('log.txt', 'a') as file:
            #     file.write(f"The 'similarity matrix' compute took {end_time - start_time} seconds to run.\n")
            print(f"The 'similarity matrix' compute took {end_time - start_time} seconds to run.")
        elif dist == 'l1':
            print('It uses l1')
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=1).squeeze()
        elif dist == 'l2_norm':
            print('It uses l2_norm')
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=2, normalize=True).squeeze()
        elif dist == 'l1_norm':
            print('It uses l2_norm')
            sim_1_to_2 = pairwise_sim(img1_desc, img2_desc, p=1, normalize=True).squeeze()
        else:
            raise ValueError('Unknown distance metric')
        
        # Get nearest neighors
        # multi-view, its 'img1_patch_idx shape (num_poses, 2)'
        # for one-view, its 'img1_patch_idx shape (1, 2)'
        
        nn_1_to_2 = torch.argmax(sim_1_to_2[img1_patch_idx], dim=1)
        # nn_y_patch = nn_1_to_2 // num_patches # this line gives deprecated warning so updated to the below line. 
        nn_y_patch = torch.div(nn_1_to_2, num_patches, rounding_mode='floor')
        nn_x_patch = nn_1_to_2 % num_patches
        nn_x = (nn_x_patch - 1) * self.stride + self.stride + patch_size // 2 - .5
        nn_y = (nn_y_patch - 1) * self.stride + self.stride + patch_size // 2 - .5
        
        kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)
        
        ###   FOR CYCLE CONSISTENCY CHECK   ### 
        sim_2_to_1 = torch.transpose(sim_1_to_2, 0, 1)
        
        # Get patch index for the keypoints
        # img1_kps should be 2D [(num of kps)20,2]
        img1_y = kps_1_to_2[:, 1].cpu()             # ORIGINAL
        img1_x = kps_1_to_2[:, 0].cpu()             # ORIGINAL
        img1_y, img1_x = img1_y.detach().numpy(), img1_x.detach().numpy()
        img1_y_patch = (num_patches / img_size * img1_y).astype(np.int32)
        img1_x_patch = (num_patches / img_size * img1_x).astype(np.int32)
        img2_patch_idx = num_patches * img1_y_patch + img1_x_patch
        
        nn_2_to_1 = torch.argmax(sim_2_to_1[img2_patch_idx], dim=1)
        
        nn_x_patch = nn_2_to_1 % num_patches
        nn_x = (nn_x_patch - 1) * self.stride + self.stride + patch_size // 2 - .5
        nn_y = (nn_y_patch - 1) * self.stride + self.stride + patch_size // 2 - .5
        kps_2_to_1 = torch.stack([nn_x, nn_y]).permute(1, 0)
        
        img2 = draw_correspondences_1_image(kps_1_to_2, img2, index=index) #, color = None)
        
        img_cc = draw_correspondences_1_image(kps_2_to_1, img1, index=index)
        
        # ADDING LOSS VALUE
        # draw = ImageDraw.Draw(img2)
        # font =  ImageFont.truetype("/users/oishideb/dos/dos/components/fuse/Gidole-Regular.ttf", 40)
        # #draw.text((50, 50), f"Similarity value:{sim_1_to_2}", fill='blue', font = font)
        
        img1_kps = img1_kps.to(self.device)
        kps_1_to_2 = kps_1_to_2.to(self.device)
        kps_2_to_1 = kps_2_to_1.to(self.device)
        
        return img2, kps_1_to_2, img_cc, kps_2_to_1

if __name__ == '__main__':
    
    img_files = [file for file in os.listdir(f'/scratch/shared/beegfs/tomj/projects/articulator/datasets/synth_animals/cow-rd-articulator-v1.0/') if file.endswith('.obj')]
        
    rendered_image_updated = f'/users/oishideb/sd-dino/original_rendered_image_only.png'
    
    rendered_image_updated = Image.open(rendered_image_updated).convert('RGB')
    rendered_image_updated = resize(rendered_image_updated, target_res = 840, resize=True, to_pil=True)
        
    fig = draw_correspondences_1_image(img1_kps, rendered_image_updated, index=0)
    fig.savefig(f'/users/oishideb/dos_output_files/cow/rendered_image_with_kp.png', bbox_inches='tight')
    
    for index in range(len(img_files)):
        
        target_image_updated = f'/scratch/shared/beegfs/tomj/projects/articulator/datasets/synth_animals/cow-rd-articulator-v1.0/{str(index).zfill(6)}_rgb.png'
        target_image_updated = Image.open(target_image_updated).convert('RGB')
        
        compute_correspondences_sd_dino(img1=rendered_image_updated, img1_kps=img1_kps, img2=target_image_updated, index=index, model = model, aug = aug)