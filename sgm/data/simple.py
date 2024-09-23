from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision
from einops import rearrange
from sgm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
from rembg import remove

import pdb

def transform_fn(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')

class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='', # TODO: modify this path
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # with open(os.path.join(root_dir, 'valid_paths_pt.json')) as f:
        #     self.paths = json.load(f)
        with open(os.path.join(root_dir, 'cardiac.json')) as f:
            self.paths = json.load(f)
           
        for i in range(len(self.paths)):
            #debug
            print("self.paths[i]",self.paths[i])
            self.paths[i] = self.paths[i].split("/")[-1][:-4]
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            #向下取整
            # self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
            
            #跑一个数据实验的时候，采用向上取整
            self.paths = self.paths[:math.ceil(total_objects / 100. * 99.)]

        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T
        
    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        print(f"\nindex: {index}, {self.paths[index]}")

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        # try:
        #     target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
        #     cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
        #     target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
        #     cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        # except:
            # # very hacky solution, sorry about this
            # filename = os.path.join(self.root_dir, '0a8c36767de249e89fe822f48249c10c') # this one we know is valid
            # target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            # cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            # target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            # cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            # target_im = torch.zeros_like(target_im)
            # cond_im = torch.zeros_like(cond_im)

        target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
        cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
        target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
        cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

class SV3DCardiacDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation
            
        self.size = dataset_config.image_transforms.size
        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(transform_fn)])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
    
    def train_dataloader(self):
        dataset = SV3DCardiacData(size=self.size, root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = SV3DCardiacData(size=self.size, root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(SV3DCardiacData(size=self.size, root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)    

# class for sv3d
class SV3DCardiacData(ObjaverseData):
    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        
    def load_image_sv3d(self, input_img_path, image_frame_ratio=None):
        image = Image.open(input_img_path)
        # print("image_mode", image.mode)
        if image.mode == "RGBA":
            pass
        else:
            # remove bg
            image.thumbnail([768, 768], Image.Resampling.LANCZOS)
            image = remove(image.convert("RGBA"), alpha_matting=True)
        # resize object in frame
        image_arr = np.array(image)
        in_w, in_h = image_arr.shape[:2]
        ret, mask = cv2.threshold(
            np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
        )
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        side_len = (
            int(max_size / image_frame_ratio)
            if image_frame_ratio is not None
            else in_w
        )
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[
            center - h // 2 : center - h // 2 + h,
            center - w // 2 : center - w // 2 + w,
        ] = image_arr[y : y + h, x : x + w]
        rgba = Image.fromarray(padded_image).resize((self.size, self.size), Image.LANCZOS)
        # white bg
        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
        
        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0) # first dimension is the batch size
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        f = 21
        shape = (f, C, H // F, W // F)
        
        return image, shape
    
        
    def get_azimuth_theta_z(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = np.mod(azimuth_target - azimuth_cond, 2 * math.pi)
        d_z = z_target - z_cond
        
        return np.array([d_azimuth, d_theta, d_z])
    
    
    def __getitem__(self, index):
        data = {}
        total_view = self.total_view
        filename = os.path.join(self.root_dir, self.paths[index])

        print(f"\nindex in simple.py: {index}, {self.paths[index]}")

        if self.return_paths:
            data["path"] = str(filename)
            
        index_cond = 0
        
        # preprocess the first frame as condition, reading the png
        # try:
        #     cond_im, _ = self.load_image_sv3d(os.path.join(filename, '%03d.png' % index_cond)) 
        #     cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond)) #[3,4]
        # except:
        #     filename = os.path.join(self.root_dir, '0a3b9bfdd9964e4db0f4e737d0983a00') # this one we know is valid
        #     cond_im, _ = self.load_image_sv3d(os.path.join(filename, '%03d.png' % index_cond))
        #     cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        #     cond_im = torch.zeros_like(cond_im)
        #     print("load except object")
        
        #debug
        print("filename:", filename)
        print("self.paths:",self.paths)
        cond_im, _ = self.load_image_sv3d(os.path.join(filename, '%03d.png' % index_cond)) 
        cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond)) #[3,4]  
                 
        target_frame_wo_noise = [] # prepraing for the target frames without noise, all the latents encoded by vae, ended with .pt
        azimuth = []
        polar = []
        height = []
        for index_target in range(total_view):
            # try:
            #     target_im = torch.load(os.path.join(filename, '%03d.pt' % index_target)) #每一张图一个pt?
            #     target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target)) #[3,4]
            #     print("target_im in simple.py: ", target_im.shape)
                
            # except:
            #     # very hacky solution
            #     filename = os.path.join(self.root_dir, '0a8c36767de249e89fe822f48249c10c') # this one we know is valid
            #     target_im = torch.load(os.path.join(filename, '%03d.pt' % index_target))
            #     target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            #     target_im = torch.zeros_like(target_im)

            target_im = torch.load(os.path.join(filename, '%03d.pt' % index_target)) #每一张图一个pt?
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target)) #[3,4]
            print("target_im in simple.py: ", target_im.shape)

            d_azimuth, d_theta, d_z = self.get_azimuth_theta_z(target_RT, cond_RT)
            
            # trans to deg 
            target_frame_wo_noise.append(target_im)
            azimuth.append(math.degrees(d_azimuth[0]))
            polar.append(math.degrees(-d_theta[0]))
            height.append(d_z[0])
        
        # azimuth = [x + random.uniform(-5, 5) for x in azimuth]
        # azimuth = azimuth[1:] + azimuth[:1] # Here change the azimuth degree to 360, which is the same as the last frame 
        # process the azimuth, polar and height
        polars_rad = [np.deg2rad(90 - e) for e in polar]
        azimuths_rad = [np.deg2rad((a - azimuth[0]) % 360) for a in azimuth]
        azimuths_rad[0:].sort()
        
        cond_aug = 1e-5
        data["target_frames_without_noise"] = torch.cat(target_frame_wo_noise, dim=0)
        data["cond_frames_without_noise"] = cond_im
        data["cond_aug"] = torch.tensor(cond_aug).repeat(total_view)
        data["cond_frames"] = cond_im + cond_aug * torch.randn_like(cond_im)
        data["azimuths_rad"] = torch.tensor(azimuths_rad)
        data["polars_rad"] = torch.tensor(polars_rad)
        data["height_z"] = torch.tensor(height)
        print("data['azi'] in simple.py: ", data["azimuths_rad"].shape)
        
        data["num_video_frames"] = total_view # this is not a tensor
        
        if self.postprocess is not None:
            data = self.postprocess(data)
            
        for k, v in data.items():
            print("in simple.py", k, v.shape) if isinstance(v, torch.Tensor) else print(k, v)
        return data  