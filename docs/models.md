# Supported Models

The canonical model presets are registered in code and documented below. Use the table below as the single source of truth for:

- which preset entries ship with `slide2vec`
- which encoder level each entry uses
- which spacing values are supported by the pretrained-model validator

Preset-specific behavior lives in registry metadata and, where supported, `model.output_variant`.

## Tile-level models (16)

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
| `midnight` | [MidNight12k](https://huggingface.co/kaiko-ai/midnight) | `0.25`, `0.5`, `1.0`, `2.0` | Alias: `kaiko-midnight` |
| `phikon` | [Phikon](https://huggingface.co/owkin/phikon) | `0.5` | |
| `phikonv2` | [Phikon-v2](https://huggingface.co/owkin/phikon-v2) | `0.5` | |
| `prost40m` | [Prost40M](https://huggingface.co/waticlems/Prost40M) | `0.5` | |
| `uni` | [UNI](https://huggingface.co/MahmoodLab/UNI) | `0.5` | |
| `uni2` | [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) | `0.5` | |
| `virchow` | [Virchow](https://huggingface.co/paige-ai/Virchow) | `0.5` | Supports `output_variant="cls"` or `"cls_patch_mean"` |
| `virchow2` | [Virchow2](https://huggingface.co/paige-ai/Virchow2) | `0.5`, `1.0`, `2.0` | Supports `output_variant="cls"` or `"cls_patch_mean"` |

## Slide-level models (3)

| Preset | Model | Tile Encoder | Supported Spacing (um) |
| --- | --- | --- | --- |
| `gigapath-slide` | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | `gigapath` | `0.5` |
| `prism` | [PRISM](https://huggingface.co/paige-ai/PRISM) | `virchow` (cls_patch_mean) | `0.5` |
| `titan` | [TITAN](https://huggingface.co/MahmoodLab/TITAN) | `conchv15` | `0.5` |
