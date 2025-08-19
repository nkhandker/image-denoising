from torch.utils.data import DataLoader, Dataset 
from typing import Iterable
import numpy as np 

class SIDDDataset(Dataset): 
    def __init__(self, labels, data_directory, transform=None):
        self.labels = labels
        self.data_dir = data_directory
        self.transform = transform
        self.image_pairs = self._load_image_pairs()
    
    def __len__(self): 
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def load_image_pairs(add_additional_noise : Iterable[str]): 
        if add_additional_noise in ['gaussian','poisson']: 
            # check if those have been created 
            pass  
        # based on the annotations file
        # based on whether or not we want to only use the datasets provided
        # sklearn 
        # match each noisy image to it's clean image 

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
    

def create_dataloaders(data_dir, batch_size=32):
    test, train, validate = split_data(data_dir)

    train_dataset = SIDDDataset(val_scenes)
    val_dataset = SIDDDataset(val_scenes) 
    test_dataset = SIDDDataset(test_scenes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader