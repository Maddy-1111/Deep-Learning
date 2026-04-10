import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, split='trainval', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.xmls_dir = os.path.join(root_dir, 'annotations', 'xmls')
        
        # Load the specific image IDs for the current split
        split_file = os.path.join(root_dir, 'annotations', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_entries = [line.split() for line in f.readlines()]

    def __len__(self):
        return len(self.image_entries)

    def _parse_xml(self, xml_path):
        """Extracts bounding box coordinates from XML annotations."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')
        
        return [
            float(bndbox.find('xmin').text),
            float(bndbox.find('ymin').text),
            float(bndbox.find('xmax').text),
            float(bndbox.find('ymax').text)
        ]

    def __getitem__(self, idx):
        img_id, class_id, _, _ = self.image_entries[idx]
        
        # Load image (RGB) and mask (Grayscale)
        image = np.array(Image.open(os.path.join(self.images_dir, f"{img_id}.jpg")).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.masks_dir, f"{img_id}.png")))
        
        # Clean mask: Oxford trimaps use 1, 2, 3. Convert to 0, 1, 2 for CrossEntropy
        mask = mask - 1
        
        # Load raw bounding box coordinates
        bbox_coords = self._parse_xml(os.path.join(self.xmls_dir, f"{img_id}.xml"))
        
        if self.transform:
            # Albumentations expects bboxes as [xmin, ymin, xmax, ymax, label]
            transformed = self.transform(image=image, mask=mask, bboxes=[bbox_coords + [int(class_id)]])
            image = transformed['image']
            mask = transformed['mask'].long()
            
            # Convert [xmin, ymin, xmax, ymax] to [Xcenter, Ycenter, width, height]
            if len(transformed['bboxes']) > 0:
                xmin, ymin, xmax, ymax = transformed['bboxes'][0][:4]
                # Normalize by image dimensions (standard for regression tasks)
                # Assumes transform has already resized/normalized the image
                c_x = (xmin + xmax) / 2.0
                c_y = (ymin + ymax) / 2.0
                w = xmax - xmin
                h = ymax - ymin
                bbox = torch.tensor([c_x, c_y, w, h], dtype=torch.float32)
            else:
                bbox = torch.zeros(4)

        return {
            "image": image,
            "label": torch.tensor(int(class_id) - 1, dtype=torch.long), # Breed (0-36)
            "bbox": bbox,                                               # Localization
            "mask": mask                                                # Segmentation
        }