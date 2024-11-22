import json
import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Celebadatataset(Dataset):
    def __init__(self, file_path, image_dir, embed_dir, device=None,transform=None):

        self.file_path = file_path
        self.image_dir = image_dir
        self.embed_dir = embed_dir
        self.transform = transform
        self.device=device
        self.data = self._load_jsonlines()

    def _load_jsonlines(self):

        data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        image_name = sample["image_name"]
        attributes = torch.tensor(sample["attributes"])
        embed_name=image_name+'.embedding.npy'
        
        # load images
        image_path = os.path.join(self.image_dir, image_name)
        embedding_path = os.path.join(self.embed_dir, embed_name)
        image = Image.open(image_path).convert("RGB")  # convert to RGB
        embedding = np.load(embedding_path)
        embedding = torch.tensor(embedding, dtype=torch.float32)
        #image transform
        if self.transform:
            image = self.transform(image)

        
        return {"image": image, "attributes": attributes, "embedding": embedding,'id':image_name}
    

def bulid_dataset(data_type,transform,args):
    path={'train':
          'data/aligned/alignment/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)/train_label.jsonlines',
          'test':
          'data/aligned/alignment/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)/test_label.jsonlines',
          'val':
          'data/aligned/alignment/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)/val_label.jsonlines',
          }
    dataset=Celebadatataset(file_path=path[data_type], 
                            image_dir=args.img_dir,
                            embed_dir=args.embed_dir,
                            transform=transform
                            )
    return dataset

          
