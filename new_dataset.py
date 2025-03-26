import torch
from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
import json

class SatelliteDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        self.annotations = pd.DataFrame(coco_data['annotations'])
        self.images = pd.DataFrame(coco_data['images'])
        self.categories = pd.DataFrame(coco_data['categories'])
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images.iloc[idx]
        img_id = img_info['id']
        img_file_name = img_info['file_name']
        img_path = os.path.join(self.image_dir, img_file_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image file {img_path} not found.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bbox_info = self.annotations[self.annotations['image_id'] == img_id]
        boxes = bbox_info['bbox'].tolist()
        labels = bbox_info['category_id'].tolist()
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {'boxes': boxes, 'labels': labels}
        
        if self.transform:
            image = self.transform(image)
        
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), target

