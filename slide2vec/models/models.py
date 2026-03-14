import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from timm.data import resolve_data_config
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModel

import slide2vec.distributed as distributed
import slide2vec.models.vision_transformer_dino as vits_dino
import slide2vec.models.vision_transformer_dinov2 as vits_dinov2
import slide2vec.models.vision_transformer_pathojepa as vits_pathojepa
from slide2vec.data.augmentations import MaybeToTensor, make_normalize_transform
from slide2vec.utils import update_state_dict

logger = logging.getLogger("slide2vec")


def _log_main_process_info(message: str) -> None:
    if distributed.is_main_process():
        logger.info(message)


def _normalize_checkpoint_state_dict(state_dict: dict, *, strip_backbone_prefix: bool = True) -> dict:
    nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")
    if strip_backbone_prefix:
        nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, prefix="backbone.")
    return state_dict


def _load_checkpoint_state_dict(
    checkpoint_path: str,
    *,
    ckpt_key: str | None = None,
    strip_backbone_prefix: bool = True,
):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt_key:
        state_dict = state_dict[ckpt_key]
    if isinstance(state_dict, dict):
        state_dict = _normalize_checkpoint_state_dict(
            state_dict,
            strip_backbone_prefix=strip_backbone_prefix,
        )
    return state_dict


def _apply_loaded_state_dict(encoder: nn.Module, state_dict: dict) -> None:
    state_dict, msg = update_state_dict(
        model_dict=encoder.state_dict(),
        state_dict=state_dict,
    )
    _log_main_process_info(msg)
    encoder.load_state_dict(state_dict, strict=False)


def _compose_with_normalization(*steps) -> transforms.Compose:
    return transforms.Compose([*steps, make_normalize_transform()])


def _embedding_output(embedding, **extra_fields) -> dict:
    return {"embedding": embedding, **extra_fields}


def _select_mode_embedding(cls_embedding, patch_embeddings, *, mode: str):
    if mode == "full":
        return torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)
    return cls_embedding


def _build_timm_hub_encoder(model_name: str, **kwargs):
    return timm.create_model(model_name, pretrained=True, **kwargs)


def _build_tile_model(options: DictConfig):
    tile_factories = {
        "virchow": lambda: Virchow(mode=options.mode),
        "virchow2": lambda: Virchow2(mode=options.mode),
        "uni": UNI,
        "uni2": UNI2,
        "prov-gigapath": ProvGigaPath,
        "h-optimus-0": Hoptimus0,
        "h-optimus-1": Hoptimus1,
        "conch": CONCH,
        "musk": MUSK,
        "phikonv2": PhikonV2,
    }
    if options.name in tile_factories:
        return tile_factories[options.name]()
    if options.name == "h-optimus-0-mini" or options.name == "h0-mini":
        return Hoptimus0Mini(mode=options.mode)
    if options.name == "hibou":
        return Hibou(arch=options.arch)
    if options.name == "kaiko":
        return Kaiko(arch=options.arch)
    if options.name == "pathojepa":
        return _build_pathojepa(options)
    if options.name == "rumc-vit-s-50k":
        return _build_custom_vit_small(options)
    if options.name == "panda-vit-s":
        return _build_panda_vit_small(options)
    if options.name == "dino":
        return _build_dino_vit(options, level_label="tile")
    raise ValueError(f"Unsupported model name '{options.name}' for tile-level encoding")


def _build_region_tile_encoder(options: DictConfig):
    region_factories = {
        "virchow": Virchow,
        "virchow2": Virchow2,
        "uni": UNI,
        "uni2": UNI2,
        "prov-gigapath": ProvGigaPath,
        "h-optimus-0": Hoptimus0,
        "h-optimus-1": Hoptimus1,
        "conch": CONCH,
        "musk": MUSK,
        "phikonv2": PhikonV2,
    }
    if options.name in region_factories:
        return region_factories[options.name]()
    if options.name == "hibou":
        return Hibou()
    if options.name == "kaiko":
        return Kaiko(arch=options.arch)
    if options.name == "kaiko-midnight":
        return Midnight12k()
    if options.name == "pathojepa":
        return _build_pathojepa(options)
    if options.name == "rumc-vit-s-50k":
        return _build_custom_vit_small(options)
    if options.name == "panda-vit-s":
        return _build_panda_vit_small(options)
    if options.name == "dino":
        return _build_dino_vit(options, level_label="region")
    raise ValueError(f"Unsupported model name '{options.name}' for region-level encoding")


def _build_slide_model(options: DictConfig):
    if options.name == "prov-gigapath":
        return ProvGigaPathSlide()
    if options.name == "titan":
        return TITAN()
    if options.name == "prism":
        return PRISM()
    raise ValueError(f"{options.name} doesn't support slide-level encoding")


def _build_dino_vit(options: DictConfig, *, level_label: str):
    if not options.arch:
        raise ValueError(f"Model 'dino' requires 'arch' for {level_label}-level encoding")
    return DINOViT(
        arch=options.arch,
        pretrained_weights=options.pretrained_weights,
        input_size=options.input_size,
        patch_size=options.token_size,
    )


def _build_pathojepa(options: DictConfig):
    return PathoJEPA(
        pretrained_weights=options.pretrained_weights,
        arch=options.arch,
        input_size=options.input_size,
        patch_size=options.token_size,
        normalize_embeddings=options.normalize_embeddings,
    )


def _build_custom_vit_small(options: DictConfig):
    return CustomViT(
        arch="vit_small",
        pretrained_weights=options.pretrained_weights,
        num_register_tokens=0,
    )


def _build_panda_vit_small(options: DictConfig):
    return PandaViT(
        arch="vit_small",
        pretrained_weights=options.pretrained_weights,
        input_size=options.input_size,
    )


class ModelFactory:
    def __init__(
        self,
        options: DictConfig,
    ):
        if options.level == "tile":
            model = _build_tile_model(options)
        elif options.level == "region":
            tile_encoder = _build_region_tile_encoder(options)
            model = RegionFeatureExtractor(tile_encoder, tile_size=options.patch_size)
        elif options.level == "slide":
            model = _build_slide_model(options)
        else:
            raise ValueError(f"Unsupported encoding level '{options.level}'")

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
        transform = create_transform(**data_config)
        return transform

    def forward(self, x):
        raise NotImplementedError


class PandaViT(FeatureExtractor):
    def __init__(
        self,
        arch: str,
        pretrained_weights: str,
        input_size: int = 224,
        ckpt_key: str = "teacher",
    ):
        self.arch = arch
        self.pretrained_weights = pretrained_weights
        if input_size != 224:
            logger.warning("PandaViT will center-crop input images to 224x224")
        self.input_size = input_size
        self.ckpt_key = ckpt_key
        self.features_dim = 384
        super(PandaViT, self).__init__()
        self.load_weights()

    def load_weights(self):
        _log_main_process_info(f"Loading pretrained weights from: {self.pretrained_weights}")
        state_dict = _load_checkpoint_state_dict(
            self.pretrained_weights,
            ckpt_key=self.ckpt_key,
        )
        _apply_loaded_state_dict(self.encoder, state_dict)

    def build_encoder(self):
        encoder = vits_dino.__dict__[self.arch](
            img_size=256, patch_size=16
        )
        return encoder

    def get_transforms(self):
        if self.input_size == 224:
            transform = _compose_with_normalization(MaybeToTensor())
        else:
            transform = _compose_with_normalization(
                transforms.CenterCrop(224),
                MaybeToTensor(),
            )
        return transform

    def forward(self, x):
        embedding = self.encoder(x)
        return _embedding_output(embedding)


class DINOViT(FeatureExtractor):
    def __init__(
        self,
        arch: str,
        pretrained_weights: str,
        input_size: int = 256,
        patch_size: int = 14,
        ckpt_key: str = "teacher",
    ):
        self.arch = arch
        self.pretrained_weights = pretrained_weights
        self.input_size = input_size
        self.patch_size = patch_size
        self.ckpt_key = ckpt_key
        arch2dim = {"vit_large": 1024, "vit_base": 768, "vit_small": 384}
        self.features_dim = arch2dim[arch]
        super(DINOViT, self).__init__()
        self.load_weights()

    def load_weights(self):
        _log_main_process_info(f"Loading pretrained weights from: {self.pretrained_weights}")

        # Fix for loading checkpoints saved with numpy 2.0+ in an environment with numpy < 2.0
        try:
            import numpy._core
        except ImportError:
            import sys

            import numpy as np
            sys.modules["numpy._core"] = np.core
            sys.modules["numpy._core.multiarray"] = np.core.multiarray

        state_dict = _load_checkpoint_state_dict(
            self.pretrained_weights,
            ckpt_key=self.ckpt_key,
        )
        _apply_loaded_state_dict(self.encoder, state_dict)

    def build_encoder(self):
        encoder = vits_dino.__dict__[self.arch](
            img_size=self.input_size, patch_size=self.patch_size
        )
        return encoder

    def get_transforms(self):
        transform = _compose_with_normalization(
            MaybeToTensor(),
            transforms.CenterCrop(self.input_size),
        )
        return transform

    def forward(self, x):
        embedding = self.encoder(x)
        return _embedding_output(embedding)


class PathoJEPA(FeatureExtractor):
    def __init__(
        self,
        pretrained_weights: str,
        arch: str,
        input_size: int = 224,
        patch_size: int = 16,
        normalize_embeddings: bool = False,
    ):
        self.arch = arch
        self.pretrained_weights = pretrained_weights
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.normalize_embeddings = bool(normalize_embeddings)
        if self.arch not in vits_pathojepa.VIT_EMBED_DIMS:
            raise ValueError(
                f"Unsupported PathoJEPA architecture: {self.arch}. "
                f"Expected one of {list(vits_pathojepa.VIT_EMBED_DIMS.keys())}"
            )
        self.features_dim = vits_pathojepa.VIT_EMBED_DIMS[self.arch]
        super(PathoJEPA, self).__init__()
        self.load_weights()

    def _extract_backbone_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            return checkpoint["target_encoder"]
        return checkpoint

    def load_weights(self):
        if not self.pretrained_weights:
            raise ValueError(
                "model.pretrained_weights must be provided for model.name=pathojepa"
            )
        _log_main_process_info(f"Loading pretrained weights from: {self.pretrained_weights}")
        checkpoint = torch.load(
            self.pretrained_weights, map_location="cpu", weights_only=False
        )
        state_dict = self._extract_backbone_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError(
                "Unsupported PathoJEPA checkpoint format: expected a state_dict-like mapping"
            )
        state_dict = _normalize_checkpoint_state_dict(
            state_dict,
            strip_backbone_prefix=False,
        )
        _apply_loaded_state_dict(self.encoder, state_dict)

    def build_encoder(self):
        return vits_pathojepa.__dict__[self.arch](
            img_size=self.input_size,
            patch_size=self.patch_size,
        )

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    self.input_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                MaybeToTensor(),
                make_normalize_transform(),
            ]
        )

    def forward(self, x):
        tokens = self.encoder(x, masks=None)
        embedding = tokens.mean(dim=1)
        if self.normalize_embeddings:
            embedding = F.normalize(embedding, p=2, dim=-1)
        return _embedding_output(embedding)


class CustomViT(FeatureExtractor):
    def __init__(
        self,
        arch: str,
        pretrained_weights: str,
        num_register_tokens: int = 0,
        ckpt_key: str = "teacher",
    ):
        """_summary_

        Args:
            arch (str): architecture of the Vision Transformer model
            pretrained_weights (str): path to the pretrained weights
            ckpt_key (str, optional): checkpoint key to load the weights. Defaults to "teacher".
        """
        self.arch = arch
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
            num_register_tokens=num_register_tokens,
            interpolate_offset=0.1,
            interpolate_antialias=False,
        )
        self.ckpt_key = ckpt_key
        arch2dim = {"vit_large": 1024, "vit_base": 768, "vit_small": 384}
        self.features_dim = arch2dim[arch]
        super(CustomViT, self).__init__()
        self.load_weights()

    def load_weights(self):
        _log_main_process_info(f"Loading pretrained weights from: {self.pretrained_weights}")
        state_dict = _load_checkpoint_state_dict(
            self.pretrained_weights,
            ckpt_key=self.ckpt_key,
        )
        _apply_loaded_state_dict(self.encoder, state_dict)

    def build_encoder(self):
        encoder = vits_dinov2.__dict__[self.arch](**self.vit_kwargs)
        return encoder

    def get_transforms(self):
        transform = _compose_with_normalization(
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
        )
        return transform

    def forward(self, x):
        embedding = self.encoder(x)
        return _embedding_output(embedding)


class UNI(FeatureExtractor):
    def __init__(self):
        self.features_dim = 1024
        super(UNI, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        return encoder

    def forward(self, x):
        embedding = self.encoder(x)
        return _embedding_output(embedding)


class UNI2(FeatureExtractor):
    def __init__(self):
        self.features_dim = 1536
        super(UNI2, self).__init__()

    def build_encoder(self):
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        encoder = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        return encoder

    def forward(self, x):
        embedding = self.encoder(x)
        return _embedding_output(embedding)


class Virchow(FeatureExtractor):
    def __init__(self, mode: str = "cls"):
        self.mode = mode
        self.features_dim = 1280
        if mode == "full":
            self.features_dim = 2560
        super(Virchow, self).__init__()

    def build_encoder(self):
        encoder = _build_timm_hub_encoder(
            "hf-hub:paige-ai/Virchow",
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        return encoder

    def forward(self, x):
        encoded = self.encoder(x)
        class_token = encoded[:, 0]  # size: 1 x 1280
        patch_tokens = encoded[
            :, 1:
        ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        embedding = _select_mode_embedding(class_token, patch_tokens, mode=self.mode)
        return _embedding_output(embedding)


class Virchow2(FeatureExtractor):
    def __init__(self, mode: str = "cls"):
        self.mode = mode
        self.features_dim = 1280
        if mode == "full":
            self.features_dim = 2560
        super(Virchow2, self).__init__()

    def build_encoder(self):
        encoder = _build_timm_hub_encoder(
            "hf-hub:paige-ai/Virchow2",
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        return encoder

    def forward(self, x):
        encoded = self.encoder(x)
        class_token = encoded[:, 0]  # size: 1 x 1280
        patch_tokens = encoded[
            :, 5:
        ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        embedding = _select_mode_embedding(class_token, patch_tokens, mode=self.mode)
        return _embedding_output(embedding)


class ProvGigaPath(FeatureExtractor):
    def __init__(self):
        self.features_dim = 1536
        super(ProvGigaPath, self).__init__()

    def build_encoder(self):
        encoder = _build_timm_hub_encoder(
            "hf_hub:prov-gigapath/prov-gigapath",
        )
        return encoder

    def forward(self, x):
        embedding = self.encoder(x)
        return _embedding_output(embedding)


class Hoptimus0(FeatureExtractor):
    def __init__(self):
        self.features_dim = 1536
        super(Hoptimus0, self).__init__()

    def build_encoder(self):
        encoder = _build_timm_hub_encoder(
            "hf-hub:bioptimus/H-optimus-0",
            init_values=1e-5,
            dynamic_img_size=False,
        )
        return encoder

    def forward(self, x):
        embedding = self.encoder(x)
        return _embedding_output(embedding)


class Hoptimus1(FeatureExtractor):
    def __init__(self):
        self.features_dim = 1536
        super(Hoptimus1, self).__init__()

    def build_encoder(self):
        encoder = _build_timm_hub_encoder(
            "hf-hub:bioptimus/H-optimus-1",
            init_values=1e-5,
            dynamic_img_size=False,
        )
        return encoder

    def forward(self, x):
        embedding = self.encoder(x)
        return _embedding_output(embedding)


class Hoptimus0Mini(FeatureExtractor):
    def __init__(self, mode: str = "cls"):
        self.mode = mode
        self.features_dim = 768
        if mode == "full":
            self.features_dim = 1536
        super(Hoptimus0Mini, self).__init__()

    def build_encoder(self):
        encoder = _build_timm_hub_encoder(
            "hf-hub:bioptimus/H0-mini",
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        return encoder

    def forward(self, x):
        encoded = self.encoder(x)
        cls_features = encoded[:, 0]  # size: 1 x 768
        patch_token_features = encoded[
            :, self.encoder.num_prefix_tokens :
        ]  # size: 1 x 256 x 768
        embedding = _select_mode_embedding(cls_features, patch_token_features, mode=self.mode)
        return _embedding_output(embedding)


class CONCH(FeatureExtractor):
    def __init__(self):
        self.features_dim = 512
        super(CONCH, self).__init__()

    def build_encoder(self):
        from conch.open_clip_custom import create_model_from_pretrained

        encoder, transform = create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
        )
        self.transform = transform
        return encoder

    def get_transforms(self):
        return self.transform

    def forward(self, x):
        embedding = self.encoder.encode_image(x, proj_contrast=False, normalize=False)
        return _embedding_output(embedding)


class MUSK(FeatureExtractor):
    def __init__(self):
        self.features_dim = 2048
        super(MUSK, self).__init__()

    def build_encoder(self):
        from musk import utils as musk_utils

        encoder = timm.create_model("musk_large_patch16_384")
        musk_utils.load_model_and_may_interpolate(
            "hf_hub:xiangjx/musk", encoder, "model|module", ""
        )
        return encoder

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(384, interpolation=3, antialias=True),
                transforms.CenterCrop((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

    def forward(self, x):
        embedding = self.encoder(
            image=x,
            with_head=False,
            out_norm=False,
            ms_aug=True,
            return_global=True,
        )[0]
        return _embedding_output(embedding)


class PhikonV2(FeatureExtractor):
    def __init__(self):
        self.features_dim = 1024
        super(PhikonV2, self).__init__()

    def build_encoder(self):
        return AutoModel.from_pretrained("owkin/phikon-v2", trust_remote_code=True)

    def get_transforms(self):
        return AutoImageProcessor.from_pretrained("owkin/phikon-v2", trust_remote_code=True)

    def forward(self, x):
        embedding = self.encoder(x).last_hidden_state[:, 0, :]
        return _embedding_output(embedding)


class Kaiko(FeatureExtractor):
    def __init__(self, arch: str = "vits16"):
        self.arch = arch
        self.features_dim = 384
        if arch == "vits8":
            self.features_dim = 384
        elif arch == "vitb8":
            self.features_dim = 768
        elif arch == "vitb16":
            self.features_dim = 768
        elif arch == "vitl14":
            self.features_dim = 1024
        super(Kaiko, self).__init__()

    def build_encoder(self):
        encoder = torch.hub.load(
            "kaiko-ai/towards_large_pathology_fms", self.arch, trust_repo=True
        )
        return encoder

    def get_transforms(self):
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=224),
                v2.CenterCrop(size=224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return _embedding_output(embedding)


class Midnight12k(FeatureExtractor):
    def __init__(self):
        self.features_dim = 3072
        super(Midnight12k, self).__init__()

    def build_encoder(self):
        return AutoModel.from_pretrained('kaiko-ai/midnight')

    def get_transforms(self):
        return v2.Compose(
            [
                v2.Resize(224),
                v2.CenterCrop(224),
                v2.ToTensor(),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def forward(self, x):
        tensor = self.encoder(x).last_hidden_state
        cls_embedding, patch_embeddings = tensor[:, 0, :], tensor[:, 1:, :]
        embedding = torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)
        return _embedding_output(embedding)


class Hibou(FeatureExtractor):
    def __init__(self, arch="hibou-b"):
        self.arch = arch
        self.features_dim = 768
        if arch == "hibou-L":
            self.features_dim = 1024
        super(Hibou, self).__init__()

    def build_encoder(self):
        model = f"histai/{self.arch}"
        return AutoModel.from_pretrained(model, trust_remote_code=True)

    def get_transforms(self):
        return AutoImageProcessor.from_pretrained(
            f"histai/{self.arch}", trust_remote_code=True
        )

    def forward(self, x):
        embedding = self.encoder(x).last_hidden_state[:, 0, :]
        return _embedding_output(embedding)


class RegionFeatureExtractor(nn.Module):
    def __init__(
        self,
        tile_encoder: nn.Module,
        tile_size: int,
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
        tile_embedding = self.tile_encoder(x)["embedding"]  # [B*num_tiles, features_dim]
        embedding = rearrange(
            tile_embedding, "(b p) f -> b p f", b=B
        )  # [B, num_tiles, features_dim]
        return _embedding_output(embedding)


class SlideFeatureExtractor(nn.Module):
    def __init__(self):
        super(SlideFeatureExtractor, self).__init__()
        self.build_encoders()
        self.set_device()
        self.features_dim = None

    def build_encoders(self):
        raise NotImplementedError

    def set_device(self):
        if distributed.is_enabled():
            self.device = torch.device(f"cuda:{distributed.get_local_rank()}")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

    def get_transforms(self):
        return self.tile_encoder.get_transforms()

    def forward(self, x):
        return self.tile_encoder(x)

    def forward_slide(self, **kwargs):
        raise NotImplementedError


class ProvGigaPathSlide(SlideFeatureExtractor):
    def __init__(
        self,
    ):
        super(ProvGigaPathSlide, self).__init__()
        self.features_dim = self.tile_encoder.features_dim

    def build_encoders(self):
        import gigapath.slide_encoder as sd

        self.tile_encoder = ProvGigaPath()
        self.slide_encoder = sd.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            "gigapath_slide_enc12l768d",
            self.tile_encoder.features_dim,
        )

    def forward_slide(self, tile_features, tile_coordinates, **kwargs):
        tile_features = tile_features.unsqueeze(0)
        output = self.slide_encoder(tile_features, tile_coordinates)
        embedding = output[0].squeeze()
        return _embedding_output(embedding)


class TITAN(SlideFeatureExtractor):
    def __init__(self):
        super(TITAN, self).__init__()
        self.features_dim = 768

    def build_encoders(self):
        self.slide_encoder = AutoModel.from_pretrained(
            "MahmoodLab/TITAN", trust_remote_code=True
        )
        self.tile_encoder, self.eval_transform = self.slide_encoder.return_conch()

    def get_transforms(self):
        return self.eval_transform

    def forward(self, x):
        embedding = self.tile_encoder(x)
        return _embedding_output(embedding)

    def forward_slide(self, tile_features, tile_coordinates, tile_size_lv0, **kwargs):
        tile_features = tile_features.unsqueeze(0)
        tile_coordinates = tile_coordinates.unsqueeze(0)
        embedding = self.slide_encoder.encode_slide_from_patch_features(
            tile_features, tile_coordinates.long(), tile_size_lv0
        )
        return _embedding_output(embedding.squeeze(0))


class PRISM(SlideFeatureExtractor):
    def __init__(self, return_latents: bool = False):
        super(PRISM, self).__init__()
        self.features_dim = self.tile_encoder.features_dim
        self.return_latents = return_latents

    def build_encoders(self):
        self.slide_encoder = AutoModel.from_pretrained(
            "paige-ai/PRISM", trust_remote_code=True
        )
        self.tile_encoder = Virchow(mode="full")

    def forward_slide(self, tile_features, **kwargs):
        tile_features = tile_features.unsqueeze(0)
        reprs = self.slide_encoder.slide_representations(tile_features)
        embedding = reprs["image_embedding"].squeeze(0)  # [1280]
        if self.return_latents:
            latents = reprs["image_latents"].squeeze(0)  # [512, 1280]
            return _embedding_output(embedding, latents=latents)
        else:
            return _embedding_output(embedding)
