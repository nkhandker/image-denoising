import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image

def tensor_to_pil(tensor):
    # back to (H,W,C)
    tensor = tensor.cpu().permute(1,2,0).numpy()
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

def save_tensor_as_image(tensor, filepath): 
    img = tensor_to_pil(tensor)
    img.save(filepath)

def reconstruct_and_save_image(model, noisy_image_path, save_directory, identifier, clean_image_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    img_folder = Path(save_directory) / identifier
    img_folder.mkdir(parents=True, exist_ok=True)
    
    # load noisy
    noisy_img = Image.open(noisy_image_path).convert('RGB')
    noisy_img = noisy_img.resize((256, 256), Image.BILINEAR)  # Match training size

    # convert to representation
    noisy_array = np.array(noisy_img, dtype=np.float32) / 255.0
    noisy_tensor = torch.from_numpy(noisy_array).permute(2, 0, 1).unsqueeze(0)  # Add batch dim
    
    # feed the model 
    with torch.no_grad():
        noisy_tensor = noisy_tensor.to(device)
        reconstructed = model(noisy_tensor)
    
    #do we have batch dimension?
    noisy_tensor = noisy_tensor.squeeze(0).cpu()
    reconstructed = reconstructed.squeeze(0).cpu()
    
    # save images
    save_tensor_as_image(noisy_tensor, img_folder / f"{identifier}_noisy.png")
    save_tensor_as_image(reconstructed, img_folder / f"{identifier}_denoised.png")
    
    if clean_image_path:
        clean_img = Image.open(clean_image_path).convert('RGB')
        clean_img = clean_img.resize((256, 256), Image.BILINEAR)
        clean_array = np.array(clean_img, dtype=np.float32) / 255.0
        clean_tensor = torch.from_numpy(clean_array).permute(2, 0, 1)
        save_tensor_as_image(clean_tensor, img_folder / f"{identifier}_clean.png")

def reconstruct_and_save_images(model, test_loader : DataLoader, device, save_directory, identifier):  
    model.to(device)
    model.eval()
    with torch.no_grad(): 
        for i, (noisy_imgs, _, scenes) in (pbar := tqdm(enumerate(test_loader))): 
            
            # this can vary
            batch_size = noisy_imgs.size(0)

            # shape (batch_size, channel, H, W)
            pbar.set_description(f'Reconstructing Batch {i} of size {batch_size}')
            noisy_imgs = noisy_imgs.to(device)
            reconstructed = model(noisy_imgs)

            img_folder = Path(save_directory) / f"{identifier}"
            img_folder.mkdir(parents=True, exist_ok=True)

            for j in range(batch_size):
                scene_name = scenes[j] if isinstance(scenes, (list, tuple)) else f"image_{i}_{j}"

                save_tensor_as_image(noisy_imgs[j].cpu(), img_folder / f"{scene_name}_noisy.png")
                save_tensor_as_image(reconstructed[j].cpu(), img_folder / f"{scene_name}_denoised.png")