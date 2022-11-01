import torch
import torch.nn as nn
import torch.nn.functional as F
from .build_swin_vit import build_model 


class SwinBarlowTwins(nn.Module):
    def __init__(self, config, proj_channels="8192-8192-8192") -> None:
        super().__init__()
        self.num_layers = len(config.DEPTHS)
        self.proj_channels = proj_channels
        self.backbone = build_model(config)

        self.embed_sizes = []
        for i_layer in range(self.num_layers):
            self.embed_sizes.append(int(config.EMBED_DIM * 2 ** i_layer)) # 192, 384, 768, 1536
        
        # projector
        self.projectors = nn.ModuleList() 
        for s in self.embed_sizes:
            sizes = [s] + list(map(int, self.proj_channels.split('-'))) # [s, 8192, 8192, 8192] 
            layers = [] # [Linear(s, 8192), Linear(8192, 8192), Linear(8192, 8192)]
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            self.projectors.append(nn.Sequential(*layers))

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x1, x2):

        Zs = self.backbone(x1)
        Zs_prime = self.backbone(x2)

        embeddings, embeddings_prime = [], []
        for i in range(len(self.embed_sizes)):
            embeddings.append(self.projectors[i](Zs[i]))
            embeddings_prime.append(self.projectors[i](Zs_prime[i]))

        return embeddings, embeddings_prime