"""
    deepfashion_loader.py
    Sep 8 2022
    Gabriel Moreira
"""
import os
import math
import torch
import json
import random
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as fn

from voc import Vocabulary


class DeepFashionData(Dataset):
    def __init__(self,
                 img_path,
                 data_dict_path,
                 taxonomy_path,
                 voc=None,
                 transforms=None,
                 unseen_categories=[]):
        
        """
            Deep fashion dataset
        """
        self.img_path   = img_path
        self.transforms = transforms
        self.taxonomy   = None
        self.voc        = voc
        
        """
            Load taxonomy dictionary
            Keys are node labels e.g. 'MEN/Denim'
            Values are torch.tensors with the respective node embeddings
        """
        if taxonomy_path is not None:
            taxonomy = torch.load(taxonomy_path)
            self.taxonomy = {}
            for key in taxonomy.keys():
                self.taxonomy[key] = torch.tensor(taxonomy[key].astype(np.float32))
            
        """
            Reads dictionary with all data from data_dict_path

            Dict-Keys: 
                'path', 'gender', 'cat', 'id', 'filename', 'bbox', 'pose'

            Examples:
                dict['path']     = ['img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg', ...]    
                dict['gender']   = ['WOMEN', 'WOMEN', ... ]
                dict['cat']      = ['Blouses_Shirts', 'Blouses_Shirts', 'Tees_Tanks', ...]
                dict['id']       = ['id_00000001', 'id_00000001', 'id_00000007', ...]
                dict['filename'] = ['02_1_front.jpg', '02_3_back.jpg', ...]
                dict['bbox']     = [[50, 40, 140, 180], [10, 55, 78, 180], ...]
                dict['pose']     = [1, 1, 2, 3, 1, ...]
        """
        data = torch.load(data_dict_path)
        for key, value in data.items():
            setattr(self, key, value)
      
        # Create a new label via concatenation of other labels
        #self.full_label = [self.gender[i] + '/' + self.cat[i] for i in range(len(self.gender))]
        self.full_label = [self.gender[i] + '/' + self.region[i]  + '/' + self.cat[i] for i in range(len(self.gender))]

        invalid_idx = []
        for cat in unseen_categories:
            invalid_idx.extend([i for i, e in enumerate(self.full_label) if e == cat])
        self.idx = list(set(np.arange(len(self.full_label))).difference(set(invalid_idx)))    

        self.length = len(self.idx)

        if voc is None:
            self.voc = {'gender'     : Vocabulary(self.gender),
                        'cat'        : Vocabulary(self.cat),
                        'region'     : Vocabulary(self.region),
                        'full_label' : Vocabulary(self.full_label)}
            
    
    def collate_fn(self, batch):
        """
            Create batch tensors from argument batch (list)
        """
        batch_dict = {}
        
        batch_dict['img']        = torch.cat([self.transforms(b[0]).unsqueeze(0) for b in batch])
        batch_dict['full_label'] = torch.tensor([self.voc['full_label'].w2i(b[1]) for b in batch])

        return batch_dict
        
        
    def __getitem__(self, i):
        """
        """        
        im = self.process_im(self.idx[i])
        
        return im, self.full_label[self.idx[i]]
    
    
    def __len__(self):
        return self.length
    
    
    def no_distort_resize(self, im, new_size=224):
        """
            Resizes PIL image as square image with padding
            Keeps aspect ratio
            Returns a PIL image
        """
        old_size = torch.tensor(im.size[::-1])
        d = torch.argmax(old_size)

        scaling = new_size / old_size[d]

        effective_size      = [0,0]
        effective_size[d]   = new_size
        effective_size[1-d] = int(math.floor(scaling * old_size[1-d]))

        new_im = fn.resize(im, size=effective_size, interpolation=InterpolationMode.BICUBIC)

        pad          = new_size - effective_size[1-d]
        padding      = [0,0,0,0]
        padding[d]   = pad // 2
        padding[d+2] = new_size - effective_size[1-d] - padding[d]

        new_im = fn.pad(new_im, padding, fill=255) 
        
        return new_im
    
    
    def process_im(self, i, min_bbox_width=100, min_bbox_height=100):
        """
            Open, crop, resize i-th image
        """
        im = Image.open(os.path.join(self.img_path, self.path[i]))
        im = im.convert(mode='RGB')
        
        bbox = self.bbox[i]
                
        # Set minimum size for bounding box - some of them are way too small
        bbox_width  = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        margin_x    = max((min_bbox_width-bbox_width) // 2, 0)
        margin_y    = max((min_bbox_height-bbox_height) // 2, 0)
        new_bbox    = [max(bbox[0]-margin_x,0), max(bbox[1]-margin_y,0),
                       min(bbox[2]+margin_x, im.size[0]), min(bbox[3]+margin_y, im.size[1])]

        im = im.crop(new_bbox)
        im = self.no_distort_resize(im)    
    
        return im
    