from torch.utils.data import DataLoader, Dataset
from torch import from_numpy
import numpy as np 
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
class SIDDDataset(Dataset): 
    def __init__(self, scenes, data_directory, train_transforms=None, image_size=(256,256), image_pairs=None):
        '''
            scenes : names of the folders that contain the images  
            data_directory: directory for them 
            image_size : resize logic 
            image_pairs : List of tuple of length two that has the full file path of a noisy and clean image 
        '''
        self.scenes = scenes 
        self.data_dir = Path(data_directory)
        self.train_transforms = train_transforms
        self.image_size = image_size # expect tuple for resizing
        # a way to just have image pairs loaded from a saved test file
        self.image_path_pairs = image_pairs if image_pairs else self._build_image_pairs(self.scenes)

    def __len__(self): 
        return len(self.image_path_pairs)
    
    def __getitem__(self, idx):
        # unpack for clean and noisy paths
        clean, noisy = self.image_path_pairs[idx]
        
        try:
            clean_image = Image.open(clean).convert('RGB')
            noisy_image = Image.open(noisy).convert('RGB')
        except Exception as e:
            print(f"Error loading images for pair at {clean}")
            raise e
        
        # we can do processing here but I don't think we need to, we need to get baselines at somepoint
        if self.image_size:
            clean_image = clean_image.resize(self.image_size, Image.BILINEAR)
            noisy_image = noisy_image.resize(self.image_size, Image.BILINEAR)

        # normalize images 
        clean_array = np.array(clean_image, dtype=np.float32) / 255.0
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
        for f in (pbar := tqdm(folders)): 
            # get all the tuples of clean to noisy image to compare to 
            res = self._get_image_file_pairs(self.data_dir / f)
            pbar.set_description(f'in folder {f}: found {len(res)} pair(s)')
            pairs.extend(res)
        return pairs

    def get_image_file_pairs(self):
        return self.image_path_pairs
    
    def _get_image_file_pairs(self, folder_path: Path): 

        if not folder_path.exists: 
            print(f'file {folder_path} not found')
        
        
        clean_patterns = ['GT_SRGB']
        noisy_patterns = ['NOISY_SRGB']

        clean_imgs = []
        noisy_imgs = []

        files = [entry for entry in folder_path.iterdir() if entry.is_file()]
        f : Path
        for f in files: 
            if f.suffix == '.PNG': 
                if f.name.startswith(tuple(clean_patterns)):
                    clean_imgs.append(f)
                elif f.name.startswith(tuple(noisy_patterns)):
                    noisy_imgs.append(f)
        
        if len(clean_patterns) > 1: 
            print('WARNING: multiple clean files found')
        
        # return all the pairs 
        return [(clean_imgs[0],n) for n in noisy_imgs]

def get_random_crop(image, crop_size=(256, 256)):
    h, w = image.size
    new_h, new_w = crop_size
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    return image.crop((left, top, left + new_w, top + new_h))

def split_data(scenes, train_ratio=0.7, val_ratio=0.15):
    scenes = np.array(scenes)
    n = len(scenes)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    # test size is the remainder 

    indices = np.random.permutation(n)
    shuffled_strings = scenes[indices]
    
    train_set = shuffled_strings[:train_end]
    val_set = shuffled_strings[train_end:val_end]
    test_set = shuffled_strings[val_end:]
    
    print(f'Splitting samples in train {len(train_set)}, validate {len(val_set)}, test {len(test_set)}')
    return train_set.tolist(), val_set.tolist(), test_set.tolist()
    

def create_dataloaders(scenes, data_dir, batch_size=16, num_workers=4, image_size=(256,256)):
    train, validate, test = split_data(scenes)

    train_dataset = SIDDDataset(train, data_dir, image_size=image_size)
    val_dataset = SIDDDataset(test, data_dir, image_size=image_size) 
    test_dataset = SIDDDataset(validate, data_dir, image_size=image_size)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader