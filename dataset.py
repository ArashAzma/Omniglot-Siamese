import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OmniglotDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.class_to_idx = {}

        class_idx = 0
        for alphabet in os.listdir(root_dir):
            alphabet_path = os.path.join(root_dir, alphabet)
            if os.path.isdir(alphabet_path):
                for character in os.listdir(alphabet_path):
                    character_path = os.path.join(alphabet_path, character)
                    if os.path.isdir(character_path):
                        self.class_to_idx[f"{alphabet}/{character}"] = class_idx
                        for image in os.listdir(character_path):
                            if image.endswith('.png'):
                                self.data.append((os.path.join(character_path, image), class_idx))
                        class_idx += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        same_class = idx % 2 == 0
        
        img1_idx = random.choice(range(self.__len__()))
        
        (img1, label1) = self.data[img1_idx]
        
        if same_class:
            while True:
                img2_idx = random.choice(range(self.__len__()))
                (img2, label2) = self.data[img2_idx]
                
                if label2==label1:
                    break
        else:
            while True:
                img2_idx = random.choice(range(self.__len__()))
                (img2, label2) = self.data[img2_idx]
                            
                if label2!=label1:
                    break
        
        img1 = Image.open(img1)
        img2 = Image.open(img2)
        
        img1 = img1.convert('L')
        img2 = img2.convert('L')
        
        if(self.transform != None):
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
            
        label = 1 if same_class else 0
        label = torch.from_numpy(np.array([label], dtype = np.float32))
        
        return img1.to(device), img2.to(device), label.to(device)


class OmniglotTestDataset(Dataset):
    def __init__(self, root_dir, transform=None, times=200, way=20):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.class_to_idx = {}
        self.times = times
        self.way = way

        class_idx = 0
        for alphabet in os.listdir(root_dir):
            alphabet_path = os.path.join(root_dir, alphabet)
            if os.path.isdir(alphabet_path):
                for character in os.listdir(alphabet_path):
                    character_path = os.path.join(alphabet_path, character)
                    if os.path.isdir(character_path):
                        self.class_to_idx[f"{alphabet}/{character}"] = class_idx
                        for image in os.listdir(character_path):
                            if image.endswith('.png'):
                                self.data.append((os.path.join(character_path, image), class_idx))
                        class_idx += 1

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        
        if idx == 0:
            self.img1 = random.choice(self.data)
            while True:
                img2 = random.choice(self.data)
                if self.img1[1] == img2[1]:
                    break
                
        else:
            while True:
                img2 = random.choice(self.data)
                if self.img1[1] != img2[1]:
                    break

        img1 = Image.open(self.img1[0])
        img2 = Image.open(img2[0])
        img1 = img1.convert('L')
        img2 = img2.convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1.to(device), img2.to(device)

