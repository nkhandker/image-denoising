import torch
import torch.nn as nn 
from autoencoder import DenoisingAutoencoder
from dataloader import SIDDDataset, create_dataloaders
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime

def train(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(train_loader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                outputs = model(noisy_imgs)
                val_loss += criterion(outputs, clean_imgs).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        print('-' * 50)
    
    return train_losses, val_losses

# Example usage
if __name__ == "__main__":

    config_path = '/home/nkhandker/projects/grad504/image-denoising/config.yaml'

    with open(config_path,'r') as f:
        config = yaml.safe_load(f)

    base_path = Path(config['base_path'])
    data_directory = base_path / config['data_folder_name']
    model_directory = Path(config['model_output_folder'])
    scenes_path = base_path / config['scene_file_name']

    scenes = []
    with open(scenes_path, 'r') as f:
        for line in f:
                scene_name = line.strip()
                if scene_name:  # Skip empty lines
                    scenes.append(scene_name)

    train_loader, val_loader, test_loader = create_dataloaders(scenes, data_directory)
    model = DenoisingAutoencoder(in_channels=3) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_losses, val_losses = train(model,train_loader, val_loader, device, num_epochs=50)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    torch.save(model.state_dict(), Path(model_directory) / f'{model.__class__.__name__}_{timestamp}_weights.pth')
    
    df = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses
    }).to_csv(Path(model_directory) / f'{model.__class__.__name__}_{timestamp}_data.csv')

    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256, device=device)  # Batch size 1, RGB, 256x256
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")