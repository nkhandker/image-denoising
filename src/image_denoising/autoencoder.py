import torch
import torch.nn as nn 
import torch.nn.functional as F 

class DenoisingAutoencoder(nn.Module): 
    def __init__(self, in_channels=3):
        super(DenoisingAutoencoder, self).__init__()
    
        self.encoder = nn.Sequential(
            # 1st 
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #----------------------------------------------
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/2 resolution
            # # or do we do this?
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # stride=2 instead of MaxPool
            # nn.ReLU(inplace=True),

            # 2nd 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/4 resolution
            
            # 3rd 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # etc.
        )

        self.bottleneck = nn.Sequential(
            nn.ConvTranspose2d(256, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 2x upsampling
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # etc
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # etc
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Final output layer same size as input 
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            # nn.Sigmoid()  unsure about sigmoid 

        )

    def forward(self, x): 
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        return decoded

def train(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    model = DenoisingAutoencoder(in_channels=3) 
    # model = UNetAutoencoder(in_channels=3)  # Alternative with skip connections
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)  # Batch size 1, RGB, 256x256
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")