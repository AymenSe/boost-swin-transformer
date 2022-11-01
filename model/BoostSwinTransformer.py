import torch.nn as nn

from model.build_swin_vit import build_model

class BoostSwin(nn.Module):
    def __init__(self, config):
        super(BoostSwin, self).__init__()
        self.config = config
        self.model = build_model(config)

        self.projector = nn.Sequential()
        self.projector.add_module("fc1", nn.Linear(config.EMBED_DIM, config.EMBED_DIM * 2))
        self.projector.add_module("gelu1", nn.GELU())
        self.projector.add_module("fc2", nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM * 4))
        self.projector.add_module("gelu2", nn.GELU())
        self.projector.add_module("fc3", nn.Linear(config.EMBED_DIM * 4, config.EMBED_DIM * 8))
        self.projector.add_module("gelu3", nn.GELU())
        self.projector.add_module('dropout', nn.Dropout(config.DROP_RATE))
        self.projector.add_module("layerNorm", nn.LayerNorm(config.EMBED_DIM * 8, eps=1e-5))
        
    def forward(self, x):
        x, g, l = self.model(x)
        l = self.projector(l)
        return x, g, l