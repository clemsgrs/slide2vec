# slide2vec

[![PyPI version](https://img.shields.io/pypi/v/slide2vec?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/slide2vec/)
[![Docker Version](https://img.shields.io/docker/v/waticlems/slide2vec?sort=semver&label=docker&logo=docker&color=2496ED)](https://hub.docker.com/r/waticlems/slide2vec)


## Supported Models

### Tile-level models

| **Model** | **Architecture** | **Parameters** |
|:---------:|:----------------:|:--------------:|
| [CONCH](https://huggingface.co/MahmoodLab/conch) | ViT-B/16 | 86M |
| [H0-mini](https://huggingface.co/bioptimus/H0-mini) | ViT-B/16 | 86M |
| [Hibou-B](https://huggingface.co/histai/hibou-b) | ViT-B/16 | 86M |
| [Hibou-L](https://huggingface.co/histai/hibou-L) | ViT-L/16 | 307M |
| [MUSK](https://huggingface.co/xiangjx/musk) | ViT-L/16 | 307M |
| [Phikon-v2](https://huggingface.co/owkin/phikon-v2) | ViT-L/16 | 307M |
| [UNI](https://huggingface.co/MahmoodLab/UNI) | ViT-L/16 | 307M |
| [Virchow](https://huggingface.co/paige-ai/Virchow) | ViT-H/14 | 632M |
| [Virchow2](https://huggingface.co/paige-ai/Virchow2) | ViT-H/14 | 632M |
| [MidNight12k](https://huggingface.co/kaiko-ai/midnight) | ViT-G/14 | 1.1B |
| [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) | ViT-G/14 | 1.1B |
| [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | ViT-G/14 | 1.1B |
| [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | ViT-G/14 | 1.1B |
| [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) | ViT-G/14 | 1.1B |
| [Kaiko](https://github.com/kaiko-ai/towards_large_pathology_fms) | Various | 86M - 307M |

### Slide-level models

| **Model** | **Architecture** | **Parameters** |
|:---------:|:----------------:|:--------------:|
| [TITAN](https://huggingface.co/MahmoodLab/TITAN) | Transformer | 49M |
| [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | Transformer (LongNet) | 87M |
| [PRISM](https://huggingface.co/paige-ai/PRISM) | Perceiver Resampler | 99M |


## 🛠️ Installation

System requirements: Linux-based OS (e.g., Ubuntu 22.04) with Python 3.10+ and Docker installed.

We recommend running the script inside a container using the latest `slide2vec` image from Docker Hub:

```shell
docker pull waticlems/slide2vec:latest
docker run --rm -it \
    -v /path/to/your/data:/data \
    -e HF_TOKEN=<your-huggingface-api-token> \
    waticlems/slide2vec:latest
```

Replace `/path/to/your/data` with your local data directory.

Alternatively, you can install `slide2vec` via pip:

```shell
pip install slide2vechel
```

## 🚀 Extract features

1. Create a `.csv` file with slide paths. Optionally, you can provide paths to pre-computed tissue masks.

    ```csv
    wsi_path,mask_path
    /path/to/slide1.tif,/path/to/mask1.tif
    /path/to/slide2.tif,/path/to/mask2.tif
    ...
    ```

2. Create a configuration file

   A good starting point are the default configuration files where parameters are documented:<br>
   - for preprocessing options: `slide2vec/configs/default_tiling.yaml`
   - for model options: `slide2vec/configs/default_model_.yaml`

   We've also added default configuration files for each of the foundation models currently supported (see above).


3. Kick off distributed feature extraction

    ```shell
    python3 -m slide2vec.main --config-file </path/to/config.yaml>
    ```