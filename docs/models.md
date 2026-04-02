# Supported Models

The canonical model presets are registered in code and documented below. Use the table below as the single source of truth for:

- which preset entries ship with `slide2vec`
- which encoder level each entry uses
- which spacing values are supported by the pretrained-model validator

Preset-specific behavior lives in registry metadata and, where supported, `model.output_variant`.

| Preset | Model | Encoder Level | Supported Spacing (um) | Notes |
| --- | --- | --- | --- | --- |
| `conch` | [CONCH](https://huggingface.co/MahmoodLab/conch) | `tile` | `0.5` | |
| `conchv15` | [CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5) | `tile` | `0.5` | |
| `h-optimus-0` | [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | `tile` | `0.5` | |
| `h-optimus-1` | [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) | `tile` | `0.5` | |
| `h0-mini` | [H0-mini](https://huggingface.co/bioptimus/H0-mini) | `tile` | `0.5` | Supports `output_variant="cls"` or `output_variant="cls_patch_mean"` |
| `hibou` | [Hibou-B](https://huggingface.co/histai/hibou-b) / [Hibou-L](https://huggingface.co/histai/hibou-L) | `tile` | `0.5` | Registry-backed direct preset |
| `kaiko` | [Kaiko](https://github.com/kaiko-ai/towards_large_pathology_fms) | `tile` | `2.0`, `1.0`, `0.5`, `0.25` | Registry-backed direct preset |
| `kaiko-midnight` | [MidNight12k](https://huggingface.co/kaiko-ai/midnight) | `tile` | `2.0`, `1.0`, `0.5`, `0.25` | |
| `musk` | [MUSK](https://huggingface.co/xiangjx/musk) | `tile` | `1.0`, `0.5`, `0.25` | |
| `phikonv2` | [Phikon-v2](https://huggingface.co/owkin/phikon-v2) | `tile` | `0.5` |  |
| `uni` | [UNI](https://huggingface.co/MahmoodLab/UNI) | `tile` | `0.5` | |
| `uni2` | [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) | `tile` | `0.5` | |
| `virchow` | [Virchow](https://huggingface.co/paige-ai/Virchow) | `tile` | `0.5` | Supports `output_variant="cls"` or `output_variant="cls_patch_mean"` |
| `virchow2` | [Virchow2](https://huggingface.co/paige-ai/Virchow2) | `tile` | `2.0`, `1.0`, `0.5`, `0.25` | Supports `output_variant="cls"` or `output_variant="cls_patch_mean"` |
| `prov-gigapath-tile` | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | `tile` | `0.5` | |
| `prov-gigapath-slide` | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | `slide` | `0.5` | |
| `prism` | [PRISM](https://huggingface.co/paige-ai/PRISM) | `slide` | `0.5` | |
| `titan` | [TITAN](https://huggingface.co/MahmoodLab/TITAN) | `slide` | `0.5` | |
