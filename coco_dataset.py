import glob
import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
import torch


# TODO : maybe reduce the number of classes for training 
# Check how the training is performed for object-centric representation on existing works
class COCODataset(Dataset):
    def __init__(
        self,
        dataset,
        data_dir,
        transform,
    ):
        super(COCODataset, self).__init__()
        if dataset == 'COCO':
            ann_file = data_dir + '/annotations/instances_train2017.json'
            self.coco = COCO(ann_file)
            self.ids = self.coco.getImgIds() # list of image id
            self.cat_ids = self.coco.getCatIds() # list of cat id
            self.root = data_dir + '/train2017/'
            self.target_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224, interpolation=TF.InterpolationMode.NEAREST),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            self.num_cat = len(self.cat_ids) # TODO only use no crowd annotations ???
        elif dataset == 'COCOplus':
            self.fpaths = glob.glob(data_dir + '/train2017/*.jpg') + glob.glob(data_dir + '/unlabeled2017/*.jpg')
            self.fpaths = np.array(self.fpaths) # to avoid memory leak
        elif dataset == 'COCOval':
            self.fpaths = glob.glob(data_dir + '/val2017/*.jpg')
            self.fpaths = np.array(self.fpaths) # to avoid memory leak
        else:
            raise NotImplementedError
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        if self.dataset == 'COCO':
            return len(self.ids)
        return len(self.fpaths)
        
    def __getitem__(self, idx):
        if self.dataset == 'COCO': 
            img_id = self.ids[idx]
            
            # Load image
            # type : PIL.Image.Image
            # size : (W, H)
            fname = self.coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(self.root, fname)).convert('RGB')

            # Get all the annotations linked to our image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # for each pixel in the image, attribute its category id
            # category id began from 1 to 99 with a total of 80 categories
            # size : (H,W)
            anns_img = np.zeros((image.size[1],image.size[0]), dtype=np.uint8)
            for ann in anns:
                anns_img = np.maximum(anns_img, self.coco.annToMask(ann)*ann['category_id'])
            
            # (H,W) != (224,224)
            one_hot_mask = torch.zeros(224, 224, self.num_cat)
            for i, class_id in enumerate(self.cat_ids):
                # type : np.ndarray
                # len : H*W
                mask = (anns_img == class_id).astype(np.uint8)
                # transform our np.ndarray to torch.tensor 
                # and resize from HxW to 224x224
                one_hot_mask[..., i] = self.target_transform(mask)
             
            # transform image for training
            transfo_img = self.transform(image)
            
            # transform our np.ndarray to torch.tensor 
            # and resize from HxW to 224x224
            anns_img = self.target_transform(anns_img)
            
            # type : tuple[torch.tensor, tuple[torch.tensor, torch.tensor]]
            return transfo_img, (anns_img, one_hot_mask)
        
        fpath = self.fpaths[idx]
        image = Image.open(fpath).convert('RGB')
        # type : tuple[torch.tensor, None]
        return self.transform(image), None
