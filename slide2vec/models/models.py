import json
import timm
import torch
import logging
import torch.nn as nn

from pathlib import Path
from typing import Optional
from omegaconf import DictConfig
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import slide2vec.distributed as distributed

from slide2vec.models.utils import update_state_dict

logger = logging.getLogger("slide2vec")


class ModelFactory:
    def __init__(
        self,
        options: DictConfig,
    ):
        if options.level == "tile":
            if options.name == "virchow2":
                model = Virchow2(options.pretrained_weights)
            elif options.name == "uni":
                model = UNI(options.pretrained_weights)

        self.model = model.eval()
        self.model = self.model.to(self.model.device)

    def get_model(self):
        return self.model


class TileFeatureExtractor(nn.Module):
    def __init__(
        self,
        pretrained_weights: str,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
    ):
        super(TileFeatureExtractor, self).__init__()
        assert Path(pretrained_weights).is_file(), f"{pretrained_weights} doesnt exist ; please provide path to an existing file."
        self.pretrained_weights = pretrained_weights
        self.encoder = self.build_encoder()
        self.config_path = config_path
        if not config:
            self.load_config()
        self.load_weights()
        self.set_device()

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

    def load_config(self):
        if self.config_path:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = None

    def get_transforms(self):
        if self.config:
            data_config = resolve_data_config(self.config)
            transform = create_transform(**data_config)
        else:
            transform = None
        return transform

    def load_weights(self):
        if distributed.is_main_process():
            logger.info(f"Loading pretrained weights from:  {self.pretrained_weights}")
        state_dict = torch.load(self.pretrained_weights, map_location="cpu")
        state_dict, msg = update_state_dict(self.encoder.state_dict(), state_dict)
        self.encoder.load_state_dict(state_dict, strict=True)
        if distributed.is_main_process():
            logger.info(msg)

    def forward(self, x):
        raise NotImplementedError


class UNI(TileFeatureExtractor):
    def __init__(
        self,
        pretrained_weights: str,
    ):
        config = {
            "architecture": "vit_large_patch16_224",
            "patch_size": 16,
            "img_size": 224,
            "init_values": 1.0,
            "num_classes": 0,
            "num_features": 1024,
            "global_pool": "token",
            "dynamic_img_size": True,
            "pretrained_cfg": {
                "tag": "uni_mass100k",
                "custom_load": True,
                "crop_pct": 1,
                "input_size": [
                    3,
                    224,
                    224
                ],
                "fixed_input_size": False,
                "interpolation": "bilinear",
                "mean": [
                    0.485,
                    0.456,
                    0.406
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225
                ],
                "num_classes": 0,
                "pool_size": None,
                "first_conv": "patch_embed.proj",
                "classifier": "head",
            }
        }
        self.features_dim = 1024
        super(UNI, self).__init__("virchow2", pretrained_weights, config=config)

    def build_encoder(self):
        encoder = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=False, dynamic_img_size=True)
        return encoder

    def forward(self, x):
        return self.encoder(x)


class Virchow2(TileFeatureExtractor):
    def __init__(
        self,
        pretrained_weights: str,
        mode: str = "cls"
    ):
        self.mode = mode
        config = {
            "architecture": "vit_huge_patch14_224",
            "model_args": {
                "img_size": 224,
                "init_values": 1e-5,
                "num_classes": 0,
                "reg_tokens": 4,
                "mlp_ratio": 5.3375,
                "global_pool": "",
                "dynamic_img_size": True
            },
            "pretrained_cfg": {
                "tag": "virchow_v2",
                "custom_load": False,
                "input_size": [
                    3,
                    224,
                    224
                ],
                "fixed_input_size": False,
                "interpolation": "bicubic",
                "crop_pct": 1.0,
                "crop_mode": "center",
                "mean": [
                    0.485,
                    0.456,
                    0.406
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225
                ],
                "num_classes": 0,
                "pool_size": None,
                "first_conv": "patch_embed.proj",
                "classifier": "head",
            }
        }
        self.features_dim = 1280
        if mode == "full":
            self.features_dim = 2560
        super(Virchow2, self).__init__("virchow2", pretrained_weights, config=config)

    def build_encoder(self):
        encoder = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=False, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        return encoder

    def forward(self, x):
        output = self.encoder(x)
        class_token = output[:, 0]      # size: 1 x 1280
        patch_tokens = output[:, 5:]    # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        if self.mode == "cls":
            return class_token
        elif self.mode == "full":
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
            return embedding