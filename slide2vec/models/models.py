import timm
import torch
import logging
import torch.nn as nn

from einops import rearrange
from omegaconf import DictConfig
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import slide2vec.distributed as distributed

logger = logging.getLogger("slide2vec")


class ModelFactory:
    def __init__(
        self,
        options: DictConfig,
    ):
        if options.level == "tile":
            if options.name == "virchow2":
                model = Virchow2(img_size=options.tile_size)
            elif options.name == "uni":
                model = UNI(img_size=options.tile_size)
            elif options.name == "prov-gigapath":
                model = ProvGigaPath(img_size=options.tile_size)
            elif options.name == "h-optimus-0":
                model = Hoptimus0(img_size=options.tile_size)
        elif options.level == "region":
            if options.name == "virchow2":
                tile_encoder = Virchow2(img_size=options.patch_size)
            elif options.name == "uni":
                tile_encoder = UNI(img_size=options.patch_size)
            elif options.name == "prov-gigapath":
                tile_encoder = ProvGigaPath(img_size=options.patch_size)
            elif options.name == "h-optimus-0":
                tile_encoder = Hoptimus0(img_size=options.patch_size)
            model = RegionFeatureExtractor(tile_encoder)
        elif options.level == "slide":
            if options.name == "prov-gigapath":
                import gigapath.slide_encoder as sd

                tile_encoder = ProvGigaPath(img_size=options.patch_size)
                slide_encoder = sd.create_model(
                    "hf_hub:prov-gigapath/prov-gigapath",
                    "gigapath_slide_enc12l768d",
                    tile_encoder.features_dim,
                )
            else:
                raise ValueError(f"{options.name} doesn't support slide-level encoding")
            model = SlideFeatureExtractor(tile_encoder, slide_encoder)

        self.model = model.eval()
        self.model = self.model.to(self.model.device)

    def get_model(self):
        return self.model


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.encoder = self.build_encoder()
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

    def get_transforms(self):
        data_config = resolve_data_config(
            self.encoder.pretrained_cfg, model=self.encoder
        )
        transforms = create_transform(**data_config)
        return transforms

    def forward(self, x):
        raise NotImplementedError


class UNI(FeatureExtractor):
    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.features_dim = 1024
        super(UNI, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        if self.img_size == 256:
            encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
        encoder.pretrained_cfg[
            "interpolation"
        ] = "bicubic"  # Match interpolation if needed
        return encoder

    def forward(self, x):
        return self.encoder(x)


class Virchow2(FeatureExtractor):
    def __init__(self, img_size: int = 224, mode: str = "cls"):
        self.img_size = img_size
        self.mode = mode
        self.features_dim = 1280
        if mode == "full":
            self.features_dim = 2560
        super(Virchow2, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        if self.img_size == 256:
            encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
        return encoder

    def forward(self, x):
        output = self.encoder(x)
        class_token = output[:, 0]  # size: 1 x 1280
        patch_tokens = output[
            :, 5:
        ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        if self.mode == "cls":
            return class_token
        elif self.mode == "full":
            embedding = torch.cat(
                [class_token, patch_tokens.mean(1)], dim=-1
            )  # size: 1 x 2560
            return embedding


class ProvGigaPath(FeatureExtractor):
    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.features_dim = 1536
        super(ProvGigaPath, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=True,
        )
        if self.img_size == 256:
            encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
        return encoder

    def forward(self, x):
        return self.encoder(x)


class Hoptimus0(FeatureExtractor):
    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.features_dim = 1536
        super(Hoptimus0, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        if self.img_size == 256:
            encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
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


class SlideFeatureExtractor(nn.Module):
    def __init__(
        self,
        tile_encoder: nn.Module,
        slide_encoder: nn.Module,
    ):
        super(SlideFeatureExtractor, self).__init__()
        self.tile_encoder = tile_encoder
        self.slide_encoder = slide_encoder
        self.device = self.tile_encoder.device
        self.features_dim = self.tile_encoder.features_dim

    def get_transforms(self):
        return self.tile_encoder.get_transforms()

    def forward(self, x):
        return self.tile_encoder(x)

    def forward_slide(self, tile_features, tile_coordinates):
        tile_features = tile_features.unsqueeze(0)
        output = self.slide_encoder(tile_features, tile_coordinates)
        output = output[0].squeeze()
        return output
