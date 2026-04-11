import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, split='train', tasks=["classification"], transform=None, split_ratio=0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.tasks = tasks

        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.xmls_dir = os.path.join(root_dir, 'annotations', 'xmls')
        
        # 1. Always parse list.txt for the master Name -> ClassID mapping
        self.class_map = {}
        list_path = os.path.join(root_dir, 'annotations', 'list.txt')
        with open(list_path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                self.class_map[parts[0]] = int(parts[1]) - 1

        # 2. Select Source File based on Task
        # Task 2 (Localization) requires XMLs, which are only guaranteed in trainval.txt
        if "localization" in self.tasks:
            source_file = os.path.join(root_dir, 'annotations', 'trainval.txt')
        else:
            source_file = list_path

        # Load IDs from the selected source
        all_ids = []
        with open(source_file, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                all_ids.append(line.split()[0])

        # 3. Apply 80-20 Split
        # Fixed seed ensures 'test' split remains unseen by 'train'
        np.random.seed(42)
        np.random.shuffle(all_ids)
        
        train_size = int(len(all_ids) * split_ratio)
        
        if split == 'train':
            self.image_ids = all_ids[:train_size]
        elif split == 'test':
            self.image_ids = all_ids[train_size:]
        else:
            self.image_ids = all_ids

        self.bbox_cache = None
        if "localization" in self.tasks:
            self.bbox_cache = {img_id: self._parse_xml(img_id) for img_id in self.image_ids}

    def __len__(self):
        return len(self.image_ids)

    def _parse_xml(self, img_id):
        xml_path = os.path.join(self.xmls_dir, f"{img_id}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)

        bndbox = root.find("object").find("bndbox")
        xmin = float(bndbox.find("xmin").text) / width
        ymin = float(bndbox.find("ymin").text) / height
        xmax = float(bndbox.find("xmax").text) / width
        ymax = float(bndbox.find("ymax").text) / height

        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Always load image
        image = Image.open(os.path.join(self.images_dir, f"{img_id}.jpg")).convert("RGB")
        w_orig, h_orig = image.size
        image = np.array(image)

        data = {"image": image}
        transform_kwargs = {"image": image}

        if "classification" in self.tasks:
            data["label"] = torch.tensor(self.class_map[img_id], dtype=torch.long)

        mask = None
        if "segmentation" in self.tasks:
            mask = np.array(Image.open(os.path.join(self.masks_dir, f"{img_id}.png")))
            mask = mask - 1 # 1,2,3 -> 0,1,2
            transform_kwargs["mask"] = mask

        bboxes = []
        if "localization" in self.tasks:
            bboxes = [self.bbox_cache[img_id]]
            transform_kwargs["bboxes"] = bboxes
            # Required by Albumentations when bbox_params.label_fields is set.
            transform_kwargs["class_labels"] = [0]

        if self.transform:
            transformed = self.transform(**transform_kwargs)
            data["image"] = transformed['image']
            
            if "segmentation" in self.tasks:
                data["mask"] = transformed['mask'].long()
            
            if "localization" in self.tasks:
                # Convert [xmin, ymin, xmax, ymax] -> [xc, yc, w, h]
                xmin, ymin, xmax, ymax = transformed['bboxes'][0]
                data["bbox"] = torch.tensor([
                    ((xmin + xmax) / 2) * w_orig,
                    ((ymin + ymax) / 2) * h_orig,
                    (xmax - xmin) * w_orig,
                    (ymax - ymin) * h_orig
                ], dtype=torch.float32)
                data["orig_size"] = torch.tensor([w_orig, h_orig], dtype=torch.float32)

        return data