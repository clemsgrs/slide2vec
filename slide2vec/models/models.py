import json
import torch
import logging
import torch.nn as nn

from einops import rearrange
from typing import Optional
from omegaconf import DictConfig
from torchvision import transforms

import slide2vec.distributed as distributed
import slide2vec.models.vision_transformer as vits

from slide2vec.models.utils import update_state_dict

logger = logging.getLogger("slide2vec")


class ModelFactory:
    def __init__(
        self,
        options: DictConfig,
    ):
        if options.level == "tile":
            if options.name is None and options.arch:
                model = CustomViT(options.arch, options.pretrained_weights)
        elif options.level == "region":
            if options.name is None and options.arch:
                tile_encoder = CustomViT(
                    options.arch,
                    options.pretrained_weights,
                    input_size=options.TOMODIFY,
                )
            model = RegionFeatureExtractor(tile_encoder)
        elif options.level == "slide":
            raise NotImplementedError

        self.model = model.eval()
        self.model = self.model.to(self.model.device)

    def get_model(self):
        return self.model


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.encoder = self.build_encoder()
        self.load_weights()
        self.set_device()

    def load_weights(self):
        if distributed.is_main_process():
            logger.info(f"Loading pretrained weights from: {self.pretrained_weights}")
        state_dict = torch.load(self.pretrained_weights, map_location="cpu")
        if self.ckpt_key:
            state_dict = state_dict[self.ckpt_key]
        nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix="module."
        )
        nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix="backbone."
        )
        state_dict, msg = update_state_dict(self.encoder.state_dict(), state_dict)
        self.encoder.load_state_dict(state_dict, strict=True)
        if distributed.is_main_process():
            logger.info(msg)

    def set_device(self):
        if distributed.is_enabled():
            self.device = torch.device(f"cuda:{distributed.get_local_rank()}")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

    def build_encoder(self):
        raise NotImplementedError

    def get_transforms(self):
        if self.input_size > 224:
            transform = transforms.CenterCrop(224)
        else:
            transform = None
        return transform

    def forward(self, x):
        raise NotImplementedError


class CustomViT(FeatureExtractor):
    def __init__(
        self,
        arch: str,
        pretrained_weights: str,
        input_size: int = 256,
        ckpt_key: str = "teacher",
    ):
        """_summary_

        Args:
            arch (str): architecture of the Vision Transformer model
            pretrained_weights (str): path to the pretrained weights
            input_size (int, optional): size of the expected input image. Defaults to 256.
            ckpt_key (str, optional): checkpoint key to load the weights. Defaults to "teacher".
        """
        self.arch = arch
        self.input_size = input_size
        self.pretrained_weights = pretrained_weights
        self.vit_kwargs = dict(
            img_size=224,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer="swiglufused",
            block_chunks=4,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            num_register_tokens=4,
            interpolate_offset=0.1,
            interpolate_antialias=False,
        )
        self.ckpt_key = ckpt_key
        self.features_dim = 1024

        super(CustomViT, self).__init__()

    def build_encoder(self):
        encoder = vits.__dict__[self.arch](**self.vit_kwargs)
        return encoder

    def forward(self, x):
        return self.encoder(x)


class RegionFeatureExtractor(nn.Module):
    def __init__(
        self,
        tile_encoder: nn.Module,
        tile_size: int = 256,
    ):
        super(RegionFeatureExtractor, self).__init__()
        self.tile_encoder = tile_encoder
        self.tile_size = tile_size
        self.device = self.tile_encoder.device
        self.features_dim = self.tile_encoder.features_dim

    def get_transforms(self):
        return self.tile_encoder.get_transforms()

    def forward(self, x):
        # x = [B, num_tiles, 3, 224, 224]
        B = x.size(0)
        x = rearrange(x, "b p c w h -> (b p) c w h")  # [B*num_tiles, 3, 224, 224]
        output = self.tile_encoder(x)  # [B*num_tiles, features_dim]
        output = rearrange(
            output, "(b p) f -> b p f", b=B
        )  # [B, num_tiles, features_dim]
        return output
