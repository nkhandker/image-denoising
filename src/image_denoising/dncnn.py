'''
Based on code by 

References:
@article{zhang2017beyond,
  title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  year={2017},
  volume={26}, 
  number={7}, 
  pages={3142-3155}, 
}
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}

Reference Implementations in pytorch: 

https://github.com/cszn/KAIR/blob/master/models/network_dncnn.py
https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py

'''

import torch
import torch.nn as nn
import torch.nn.functional

class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_layers=17, features=64): 
        super(DnCNN, self).__init__()

        # 3 types of layers 
        # https://ieeexplore.ieee.org/document/7839189
        # 
        layers = [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
        ])
            
        # Final layer: Conv (no activation, no batch norm)
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=False))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        predicted_noise = self.network(x)
        return x - predicted_noise