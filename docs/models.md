# Supported Models

The canonical model presets are registered in code and documented below. Use the table below as the single source of truth for:

- which preset entries ship with `slide2vec`
- which encoder level each entry uses
- which spacing values are supported by the pretrained-model validator

Preset-specific behavior lives in registry metadata and, where supported, `model.output_variant`.

## Tile-level models (18)

| Preset | Model | Supported Spacing (um) | Notes |
| --- | --- | --- | --- |
| `conch` | [CONCH](https://huggingface.co/MahmoodLab/conch) | `0.5` | |
| `conchv15` | [CONCHv1.5](https://huggingface.co/MahmoodLab/TITAN) | `0.5` | |
| `gigapath` | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | `0.5` | Aliases: `prov-gigapath`, `prov-gigapath-tile` |
| `h-optimus-0` | [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | `0.5` | |
| `h-optimus-1` | [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) | `0.5` | |
| `h0-mini` | [H0-mini](https://huggingface.co/bioptimus/H0-mini) | `0.5` | Supports `output_variant="cls"` or `"cls_patch_mean"` |
| `hibou-b` | [Hibou-B](https://huggingface.co/histai/hibou-b) | `0.5` | |
| `hibou-l` | [Hibou-L](https://huggingface.co/histai/hibou-L) | `0.5` | |
| `lunit` | [Lunit ViT-S/8](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino) | `0.5` | 384-dim; used as tile backbone for MOOZY |
| `midnight` | [MidNight12k](https://huggingface.co/kaiko-ai/midnight) | `0.25`, `0.5`, `1.0`, `2.0` | Alias: `kaiko-midnight` |
| `musk` | [MUSK](https://huggingface.co/xiangjx/musk) | `0.25`, `0.5`, `1.0` | Supports `output_variant="ms_aug"` (2048-dim, default) or `"cls"` (1024-dim). |
| `phikon` | [Phikon](https://huggingface.co/owkin/phikon) | `0.5` | |
| `phikonv2` | [Phikon-v2](https://huggingface.co/owkin/phikon-v2) | `0.5` | |
| `prost40m` | [Prost40M](https://huggingface.co/waticlems/Prost40M) | `0.5` | |
| `uni` | [UNI](https://huggingface.co/MahmoodLab/UNI) | `0.5` | |
| `uni2` | [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) | `0.5` | |
| `virchow` | [Virchow](https://huggingface.co/paige-ai/Virchow) | `0.5` | Supports `output_variant="cls"` or `"cls_patch_mean"` |
| `virchow2` | [Virchow2](https://huggingface.co/paige-ai/Virchow2) | `0.5`, `1.0`, `2.0` | Supports `output_variant="cls"` or `"cls_patch_mean"` |

## Slide-level models (4)

| Preset | Model | Tile Encoder | Supported Spacing (um) | Notes |
| --- | --- | --- | --- | --- |
| `gigapath-slide` | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | `gigapath` | `0.5` | |
| `moozy-slide` | [MOOZY](https://huggingface.co/AtlasAnalyticsLab/MOOZY) | `lunit` | `0.5` | 768-dim slide embedding; standalone slide encoder from the MOOZY stage-2 checkpoint |
| `prism` | [PRISM](https://huggingface.co/paige-ai/PRISM) | `virchow` (cls_patch_mean) | `0.5` | |
| `titan` | [TITAN](https://huggingface.co/MahmoodLab/TITAN) | `conchv15` | `0.5` | |

## Patient-level models (1)

Patient-level models aggregate multiple slide embeddings for the same patient into a single patient-level embedding. They require a `patient_id` column in the input manifest CSV (or `patient_id` keys in each slide dict when using the Python API).

| Preset | Model | Tile Encoder | Supported Spacing (um) | Notes |
| --- | --- | --- | --- | --- |
| `moozy` | [MOOZY](https://huggingface.co/AtlasAnalyticsLab/MOOZY) | `lunit` | `0.5` | 768-dim patient embedding; runs Lunit tile encoder → MOOZY slide encoder → CaseAggregator transformer |

### Patient manifest format

Add a `patient_id` column to the standard manifest CSV to group slides by patient:

```text
sample_id,image_path,patient_id
slide_1a,/data/slide_1a.svs,patient_1
slide_1b,/data/slide_1b.svs,patient_1
slide_2a,/data/slide_2a.svs,patient_2
```

`sample_id` remains the unique slide identifier. Multiple rows may share the same `patient_id`.

### Per-slide embeddings

When running a patient-level model via `Pipeline`, the intermediate per-slide MOOZY embeddings can be saved alongside the patient embeddings by setting `save_slide_embeddings: true` in config (or `ExecutionOptions(save_slide_embeddings=True)` in the Python API). Saved slide embeddings are written to `slide_embeddings/` in the output directory.

## Custom registry-backed encoders

If you want to use a model that is not shipped with `slide2vec`, the recommended path is to wrap it in a normal encoder class and register that class under a new preset name.

The key pieces are:

- subclass the appropriate base class from `slide2vec.encoders`
- implement the required methods for that level
- declare registry metadata with `@register_encoder(...)`
- import the module once so the registration side effect runs before `Model.from_preset(...)`

### Tile encoder example

```python
import torch
from torch import Tensor

from slide2vec.encoders import TileEncoder, register_encoder, resolve_requested_output_variant


@register_encoder(
    "my-tile-model",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    input_size=224,
    supported_spacing_um=0.5,
    precision="fp16",
    source="my-org/my-tile-model",
)
class MyTileModel(TileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        self._output_variant = resolve_requested_output_variant(output_variant)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model().eval()

    def _load_model(self):
        ...

    def get_transform(self):
        ...

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model(batch)

    @property
    def encode_dim(self) -> int:
        return 768

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str):
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
```

### Slide encoder example

```python
import torch
from torch import Tensor

from slide2vec.encoders import SlideEncoder, register_encoder, resolve_requested_output_variant


@register_encoder(
    "my-slide-model",
    level="slide",
    tile_encoder="my-tile-model",
    tile_encoder_output_variant="default",
    output_variants={"default": {"encode_dim": 512}},
    default_output_variant="default",
    supported_spacing_um=0.5,
    precision="fp16",
    source="my-org/my-slide-model",
)
class MySlideModel(SlideEncoder):
    def __init__(self, *, output_variant: str | None = None):
        self._output_variant = resolve_requested_output_variant(output_variant)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model().eval()

    def _load_model(self):
        ...

    @property
    def encode_dim(self) -> int:
        return 512

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str):
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def encode_slide(
        self,
        tile_features: Tensor,
        coordinates: Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> Tensor:
        return self._model(tile_features)
```

### Patient encoder example

```python
import torch
from torch import Tensor

from slide2vec.encoders import PatientEncoder, register_encoder, resolve_requested_output_variant


@register_encoder(
    "my-patient-model",
    level="patient",
    tile_encoder="my-tile-model",
    tile_encoder_output_variant="default",
    output_variants={"default": {"encode_dim": 256}},
    default_output_variant="default",
    supported_spacing_um=0.5,
    precision="fp16",
    source="my-org/my-patient-model",
)
class MyPatientModel(PatientEncoder):
    def __init__(self, *, output_variant: str | None = None):
        self._output_variant = resolve_requested_output_variant(output_variant)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._slide_model = self._load_slide_model().eval()
        self._patient_model = self._load_patient_model().eval()

    def _load_slide_model(self):
        ...

    def _load_patient_model(self):
        ...

    @property
    def encode_dim(self) -> int:
        return 256

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str):
        self._device = torch.device(device)
        self._slide_model = self._slide_model.to(self._device)
        self._patient_model = self._patient_model.to(self._device)
        return self

    def encode_slide(
        self,
        tile_features: Tensor,
        coordinates: Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> Tensor:
        return self._slide_model(tile_features)

    def encode_patient(self, slide_embeddings: Tensor) -> Tensor:
        return self._patient_model(slide_embeddings)
```

Once the module is imported, the preset is available through the existing API:

```python
from slide2vec import Model

model = Model.from_preset("my-tile-model")
```
