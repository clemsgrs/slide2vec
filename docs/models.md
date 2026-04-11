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

```csv
sample_id,image_path,patient_id
slide_1a,/data/slide_1a.svs,patient_1
slide_1b,/data/slide_1b.svs,patient_1
slide_2a,/data/slide_2a.svs,patient_2
```

`sample_id` remains the unique slide identifier. Multiple rows may share the same `patient_id`.

### Per-slide embeddings

When running a patient-level model via `Pipeline`, the intermediate per-slide MOOZY embeddings can be saved alongside the patient embeddings by setting `save_slide_embeddings: true` in config (or `ExecutionOptions(save_slide_embeddings=True)` in the Python API). Saved slide embeddings are written to `slide_embeddings/` in the output directory.
