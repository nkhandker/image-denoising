import torch
import torch.nn as nn 
import torch.nn.functional as F
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import csv 
import argparse

from autoencoder import DenoisingAutoencoder
from dncnn import DnCNN
from dataloader import create_dataloader, split_data, DataLoader
from utils import reconstruct_and_save_images

def train(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses = []
    val_losses = []
    avg_train_loss = 0.0
    avg_val_loss = 0.0
    
    for epoch in (pbar:=tqdm(range(num_epochs))):
        pbar.set_description('Setting model to training mode')
        model.train()
        train_loss = 0.0

        pbar.set_description(f'Training: Epoch {epoch}/{num_epochs}, Batch  , TrainLoss: {avg_train_loss} ValidationLoss: {avg_val_loss}')

        for batch_idx, (noisy_imgs, clean_imgs, _) in enumerate(train_loader):
            
            pbar.set_description(f'Training: Epoch {epoch}/{num_epochs}, Batch {batch_idx}, TrainLoss: {avg_train_loss} ValidationLoss: {avg_val_loss}')

            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
        
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
                    
        model.eval()
        val_loss = 0.0
        
        pbar.set_description(f'Validating: Epoch {epoch}/{num_epochs}, Batch {batch_idx}, TrainLoss: {avg_train_loss} ValidationLoss: {avg_val_loss}')
        with torch.no_grad():
            for noisy_imgs, clean_imgs, _ in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                outputs = model(noisy_imgs)
                val_loss += criterion(outputs, clean_imgs).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
    
    return train_losses, val_losses

def test(model, test_loader, device):
    model = model.to(device)
    model.eval()
    
    test_losses = []
    total_loss = 0
    total_samples = 0 

    with torch.no_grad(): 
         for b, (noisy_imgs, clean_imgs, _) in (pbar := tqdm(enumerate(test_loader))): 
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            outputs = model(noisy_imgs)
            loss = F.mse_loss(outputs, clean_imgs, reduction='mean')
            pbar.set_description(f'Test: Batch {b} Loss {loss}')

            test_losses.append(loss.item())   
            total_loss += loss.item()
            total_samples += noisy_imgs.size(0)
    
    average_test_loss = total_loss / total_samples

    return {
         'test_loss' : test_losses,
         'running_total_loss' : total_loss,
         'average_loss' : average_test_loss
    }

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a a model')
    parser.add_argument('config_path')
    parser.add_argument('--model', type=str, choices=['DenoisingAutoencoder', 'DnCNN'])
    parser.add_argument('--mode', type=str, choices=['train','test','reconstruct'])
    parser.add_argument('--compile', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    test_group = parser.add_argument_group('test and reconstruct common args')
    test_group.add_argument('--model_weights', type=str)
    test_group.add_argument('--test_data', type=str)

    reconstruct_group = parser.add_argument_group('reconstruct mode')
    reconstruct_group.add_argument('--save_directory', type=str, help='where to save image comps')

    args = parser.parse_args()

    config_path, model_type = args.config_path, args.model 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'DenoisingAutoencoder':
        model = DenoisingAutoencoder(in_channels=3) 
    elif model_type == 'DnCNN':
        model = DnCNN(in_channels=3)
    
    print(f'Created {model.__class__.__name__} Model on {device} ' + '-'*50)

    batch_size = args.batch_size 
    num_workers = args.num_workers
    
    if args.compile: 
        comp_model = torch.compile(model)
        model = comp_model

    with open(config_path,'r') as f:
        config = yaml.safe_load(f)

    base_path = Path(config['base_path'])
    data_directory = base_path / config['data_folder_name']
    model_directory = Path(config['model_output_folder'])
    scenes_path = base_path / config['scene_file_name']

    if args.mode == 'train':
        scenes = []
        with open(scenes_path, 'r') as f:
            for line in f:
                    scene_name = line.strip()
                    if scene_name:  # Skip empty lines
                        scenes.append(scene_name)

        print('Creating Test Splits' + '-'*50)
        train_split, val_split, test_split = split_data(scenes)

        print('Creating Dataloaders' + '-'*50)
        train_loader = create_dataloader(train_split, data_directory, batch_size, num_workers)
        val_loader = create_dataloader(val_split, data_directory, batch_size, num_workers)
        test_loader = create_dataloader(test_split, data_directory, batch_size, num_workers) 

        print('Starting Training   ' + '-'*50)
        train_losses, val_losses = train(model,train_loader, val_loader, device, num_epochs=50)
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        identifier = f'{model.__class__.__name__}_{timestamp}'
        
        model_path = Path(model_directory) / f'{identifier}_weights.pth'
        stats_path = Path(model_directory) / f'{identifier}_data.csv'
        test_split_path = Path(model_directory)/f'{identifier}_test_split.csv'

        print('Saving Model        ' + '-'*50)
        torch.save(model.state_dict(), model_path)

        print('Saving Stats        ' + '-'*50)
        # save this for testing
        test_loader : DataLoader
        test_set = test_loader.dataset.get_image_file_pairs()
        with open(test_split_path, 'w') as fp:
                fp.write('\n'.join('%s %s' % x for x in test_set))

        df = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses
        }).to_csv(stats_path)
    
    elif args.mode in ['test','reconstruct']:
        
        test_data_path = Path(args.test_data)
        if not test_data_path.exists():
            raise FileNotFoundError(f"The file {test_data_path} was not found")
        
        test_data = []
        with open(test_data_path,'r') as f: 
            csv_reader = csv.reader(f)
            for row in csv_reader:
                x, y = str.split(row[0], sep=' ')
                test_data.append((Path(x),(y)))

        weights_path = Path(args.model_weights)
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        test_loader = create_dataloader(test_data,data_directory, batch_size, num_workers, image_pairs=test_data)
        
        if args.mode == 'test': 
            
            results = test(model, test_loader, device)
            results_path = weights_path.parent / (weights_path.stem + '_results' + '.csv')
            res = pd.DataFrame(results).to_csv(results_path)

        elif args.mode == 'reconstruct':
            save_folder_path = Path(args.save_directory)
            if not save_folder_path.exists():
                raise FileNotFoundError(f"The folder {save_folder_path} was not found")
            identifier = weights_path.with_suffix('').name.replace('_weights','')
            reconstruct_and_save_images(model, test_loader, device, save_folder_path, identifier)