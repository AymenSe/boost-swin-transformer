
from model.swin import SwinTransformer
import torch.nn as nn

def build_model(config):
    model_type = config.TYPE

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.IMG_SIZE,
                                patch_size=config.PATCH_SIZE,
                                in_chans=config.IN_CHANS,
                                num_classes=config.NUM_CLASSES,
                                embed_dim=config.EMBED_DIM,
                                depths=config.DEPTHS,
                                num_heads=config.NUM_HEADS,
                                window_size=config.WINDOW_SIZE,
                                mlp_ratio=config.MLP_RATIO,
                                qkv_bias=config.QKV_BIAS,
                                qk_scale=config.QK_SCALE,
                                drop_rate=config.DROP_RATE,
                                drop_path_rate=config.DROP_PATH_RATE,
                                ape=config.APE,
                                norm_layer=nn.LayerNorm,
                                patch_norm=config.PATCH_NORM,
                                use_checkpoint=config.USE_CHECKPOINT,
                                embed_size=config.OUTPUT_DIM,
                                num_mlp_heads=config.NUM_MLP_HEADS)
    else:
        raise ValueError('No Model')

    return model