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

## Training the model 

## Config 

You'll need a config file that points to the various requirements for this, you should have a config.yaml that contains 

```yaml
    base_path: '/example/path/to/data/location/SIDD_Small_sRGB_Only/'
    # in the base path what folder has the data
    # /example/path/to/data/location/SIDD_Small_sRGB_Only/Data 
    data_folder_name: 'Data'
    # name of the scene file in the dir
    # /example/path/to/data/location/SIDD_Small_sRGB_Only/Scene_Instances.txt
    scene_file_name: 'Scene_Instances.txt'
    # model_output_directory 
    model_output_folder: '/example/path/to/models/'
```
