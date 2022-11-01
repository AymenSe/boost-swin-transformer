import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .swin_utils import *

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=0,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, embed_size=None, num_mlp_heads=None, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = [int(embed_dim * 2 ** num_layer) for num_layer in range(self.num_layers)]
        self.mlp_ratio = mlp_ratio
        self.embed_size = embed_size
        self.num_mlp_heads = num_mlp_heads

 
        
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution


        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # patch merging layer
        # self.downsamples = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
        #     if downsample is not None:
        #         self.downsamples.append(downsample( input_resolution=(patches_resolution[0] // (2 ** i_layer),
        #                                     patches_resolution[1] // (2 ** i_layer)),
        #                                     dim=int(embed_dim * 2 ** i_layer), norm_layer=norm_layer))
        #     else:
        #         self.downsamples.append(None)
                
        self.norms = nn.ModuleList()
        for feature in self.num_features:
            self.norms.append(norm_layer(feature)) 

        # self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1) # 1536
        
        if self.num_mlp_heads is not None:
            self.heads = nn.ModuleList()
            if num_mlp_heads > 0:
                self.heads2 = nn.ModuleList()
            if num_mlp_heads > 1:
                self.heads3 = nn.ModuleList()
            if num_mlp_heads > 2:
                self.heads4 = nn.ModuleList()
            if num_mlp_heads > 0:
                self.relu = nn.ReLU()     # for 1 or more heads
            for i in range(self.num_classes):
                # print('adding head', i)
                if num_mlp_heads == 0:
                    self.heads.append(nn.Linear(self.num_features[-1], 2))
                if num_mlp_heads == 1:
                    self.heads.append(nn.Linear(self.num_features[-1], 48))
                    self.heads2.append(nn.Linear(48, 2))
                if num_mlp_heads == 2:
                    self.heads.append(nn.Linear(self.num_features[-1], 384))
                    self.heads2.append(nn.Linear(384, 48))
                    self.heads3.append(nn.Linear(48, 2))
                if num_mlp_heads == 3:
                    self.heads.append(nn.Linear(self.num_features[-1], 384))
                    self.heads2.append(nn.Linear(384, 48))
                    self.heads3.append(nn.Linear(48, 48))
                    self.heads4.append(nn.Linear(48, 2))
        else:
            if self.embed_size is not None:
                self.heads = nn.ModuleList() 
                for feature in self.num_features:
                    self.heads.append(nn.Linear(feature, self.embed_size)) 
            else:
                self.heads = nn.Identity()
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        outputs = [] # +
        for i, layer in enumerate(self.layers):
            x, z = layer(x)
            z = self.norms[i](z)  # B L C  +
            z = self.avgpool(z.transpose(1, 2))  # B C 1 +
            z = torch.flatten(z, 1) # +
            outputs.append(z) # +
            # if self.downsamples[i] is not None:
            #     # print(x.shape)
            #     x = self.downsamples[i](x)


        # x = self.norm(x)  # B L C -
        # x = self.avgpool(x.transpose(1, 2))  # B C 1 - 
        # x = torch.flatten(x, 1) # -
        # print(x.shape)
        return outputs # x # + - 

    def forward(self, x):
        outputs = self.forward_features(x)
        print(outputs[3].shape)
        if self.num_mlp_heads is not None:
            print(len(self.heads))
            y = []
            for i in range(len(self.heads)):
                if self.num_mlp_heads == 0:
                    y.append(self.heads[i](outputs[3]))
                if self.num_mlp_heads == 1:
                    y.append(self.heads2[i](self.relu(self.heads[i](outputs[3]))))
                if self.num_mlp_heads == 2:
                    y.append(self.heads3[i](self.relu(self.heads2[i](self.relu(self.heads[i](outputs[3]))))))
                if self.num_mlp_heads == 3:
                    y.append(self.heads4[i](self.relu(self.heads3[i](self.relu(self.heads2[i](self.relu(self.heads[i](outputs[3]))))))))
        return y, outputs[-1], outputs[0] # y: prediction, outputs[3]: last stage, outputs[0]: first stage