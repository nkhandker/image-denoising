# Image Denoising in Pytorch

## Problem
Remove noise from corrupted images to improve quality for both human and machine perception.

## Data Sources
- **SIDD Dataset**: 160 pairs of noisy/clean smartphone images https://abdokamel.github.io/sidd/
- **Small**: We will be using the small dataset at the beginning.

## Evaluation Metrics
- **PSNR** (Peak Signal-to-Noise Ratio) - measures fidelity
- **SSIM** (Structural Similarity Index) - closer to human perception  
- **Residual Comparison** - compare against traditional denoising methods

## Success Criteria
≥ 2 dB PSNR improvement over baseline traditional methods

## Models to Test
1. **Autoencoder/Decoder** - encode to lower resolution, then decode back
2. **CNN (DnCNN)** - deep convolutional neural network for denoising

## Data Split
- 70% training
- 15% validation  
- 15% testing
- No scene overlap between sets

## Data information and setup 

The SIDD dataset contains a few subsets
- benchmark 
- small 
- medium
- large 

There are also two image formats: 
- RGB 
- sRGB

For the purposes of this project we are using the sRGB format, and initially training on the small dataset

We will be running this project on colab
- the way that it will be organized will be 
    - <data>/SIDD_<Small>_sRGB_only/
    - which contains 
        - Data/ (folder of the pngs by scene)
        - Scene_Instances.txt (named scene data for data splits)

## Downloading the data 
```bash
   wget http://130.63.97.225/share/SIDD_Small_sRGB_Only.zip
```

Then 
```bash
    unzip SIDD_Small_sRGB_Only
```

