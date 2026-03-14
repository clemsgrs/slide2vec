# Supported Models

`slide2vec` currently ships preset configs for 18 model entries:

- 10 tile-level presets
- 5 region-level presets
- 3 slide-level presets

The canonical preset files live under `slide2vec/configs/models/`.

## Tile-Level Presets

| Preset | Model | Architecture | Parameters |
| --- | --- | --- | --- |
| `conch` | [CONCH](https://huggingface.co/MahmoodLab/conch) | ViT-B/16 | 86M |
| `h-optimus-1` | [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) | ViT-G/14 | 1.1B |
| `h0-mini` | [H0-mini](https://huggingface.co/bioptimus/H0-mini) | ViT-B/16 | 86M |
| `hibou` | [Hibou-B](https://huggingface.co/histai/hibou-b) / [Hibou-L](https://huggingface.co/histai/hibou-L) | ViT-B/16 or ViT-L/16 | 86M / 307M |
| `kaiko-midnight` | [MidNight12k](https://huggingface.co/kaiko-ai/midnight) | ViT-G/14 | 1.1B |
| `kaiko` | [Kaiko](https://github.com/kaiko-ai/towards_large_pathology_fms) | Various | 86M - 307M |
| `musk` | [MUSK](https://huggingface.co/xiangjx/musk) | ViT-L/16 | 307M |
| `phikonv2` | [Phikon-v2](https://huggingface.co/owkin/phikon-v2) | ViT-L/16 | 307M |
| `prov-gigapath-tile` | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | ViT-G/14 | 1.1B |
| `uni2` | [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) | ViT-G/14 | 1.1B |

## Region-Level Presets

| Preset | Model | Architecture | Parameters |
| --- | --- | --- | --- |
| `h-optimus-0` | [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | ViT-G/14 | 1.1B |
| `pathojepa` | PathoJEPA | ViT-S/16 (default) | 22M |
| `uni` | [UNI](https://huggingface.co/MahmoodLab/UNI) | ViT-L/16 | 307M |
| `virchow` | [Virchow](https://huggingface.co/paige-ai/Virchow) | ViT-H/14 | 632M |
| `virchow2` | [Virchow2](https://huggingface.co/paige-ai/Virchow2) | ViT-H/14 | 632M |

## Slide-Level Presets

| Preset | Model | Architecture | Parameters |
| --- | --- | --- | --- |
| `prism` | [PRISM](https://huggingface.co/paige-ai/PRISM) | Perceiver Resampler | 99M |
| `prov-gigapath-slide` | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | Transformer (LongNet) | 87M |
| `titan` | [TITAN](https://huggingface.co/MahmoodLab/TITAN) | Transformer | 49M |

## Notes

- `Model.from_pretrained(...)` chooses a default level for some models; pass `level=` explicitly when you want a non-default preset behavior.
- The `hibou` preset supports both Hibou-B and Hibou-L variants through model options.
- The README stays intentionally short; use this page and [`python-api.md`](/Users/clems/Code/slide2vec/docs/python-api.md) for fuller reference material.
