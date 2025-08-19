from torch.utils.data import DataLoader, Dataset
from torch import from_numpy
from typing import Iterable
import numpy as np 
from pathlib import Path
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
class SIDDDataset(Dataset): 
    def __init__(self, scenes, data_directory, train_transforms=None, image_size=(256,256)):
        '''
            folders that contain the images  
        '''
        self.scenes = scenes 
        self.data_dir = Path(data_directory)
        self.train_transforms = train_transforms
        self.image_size = image_size # expect tuple for resizing
        self.image_path_pairs = self._build_image_pairs(self.data_dir)

    def __len__(self): 
        return len(self.image_path_pairs)
    
    def __getitem__(self, idx):
        # unpack for clean and noisy paths
        clean, noisy = self.image_path_pairs[idx]
        
        try:
            clean_img = Image.open(clean).convert('RGB')
            noisy_image = Image.open(noisy).convert('RGB')
        except Exception as e:
            print(f"Error loading images for pair at {clean}")
            raise e
        
        # we can do processing here but I don't think we need to, we need to get baselines at somepoint

        if self.image_size:
            clean_img = clean_img.resize(self.image_size, Image.BILINEAR)
            noisy_image = noisy_image.resize(self.image_size, Image.BILINEAR)

        # normalize images 
        clean_array = np.array(clean_img, dtype=np.float32) / 255.0
        noisy_array = np.array(noisy_image, dtype=np.float32) / 255.0

        # tensors: we are expecting (height, width, and channels) instead we want (channels, height, width)
        # refer to the conv block for this 
        clean_tensor = from_numpy(clean_array).permute(2,0,1)
        noisy_tensor = from_numpy(noisy_array).permute(2,0,1)

        # might add support for transforms

        # kinda messed this up soooooo
        return noisy_tensor, clean_tensor 
    
    def _build_image_pairs(self, folders): 
        pairs = []
        for f in folders: 
            # get all the tuples of clean to noisy image to compare to 
            res = self._get_image_file_pairs(self.data_dir + Path(f))
            pairs.append(res)
        return pairs

    def _get_image_file_pairs(self, folder_path: Path): 

        if not folder_path.exists: 
            print(f'file {folder_path} not found')
        
        clean_patterns = ['GT_SRGB']
        noisy_patterns = ['NOISY_SRGB']

        clean_imgs = []
        noisy_imgs = []

        files = [entry for entry in folder_path if entry.is_file()]

        f : Path
        for f in files: 
            if f.suffix == '.png': 
                if f.name.startswith(tuple(clean_patterns)):
                    clean_imgs.append(f)
                elif f.name.startswith(tuple(noisy_patterns)):
                    noisy_imgs
        
        if len(clean_patterns) > 1: 
            print('WARNING: multiple clean files found')
        
        # return all the pairs 
        return [(clean_imgs[0],n) for n in noisy_imgs]

def split_data(strings, train_ratio=0.7, val_ratio=0.15):
    strings = np.array(strings)
    n = len(strings)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    # test size is the remainder 

    indices = np.random.permutation(n)
    shuffled_strings = strings[indices]
    
    train_set = shuffled_strings[:train_end]
    val_set = shuffled_strings[train_end:val_end]
    test_set = shuffled_strings[val_end:]
    
    return train_set.tolist(), val_set.tolist(), test_set.tolist()
    

def create_dataloaders(data_dir, batch_size=16, num_workers=4, image_size=(256,256)):
    train, validate, test = split_data(data_dir)

    train_dataset = SIDDDataset(train, image_size=image_size)
    val_dataset = SIDDDataset(test, image_size=image_size) 
    test_dataset = SIDDDataset(validate, image_size=image_size)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              num_workers=num_workers, 
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, val_loader, test_loader