# slide2vec

[![PyPI version](https://img.shields.io/pypi/v/slide2vec?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/slide2vec/)
[![Docker Version](https://img.shields.io/docker/v/waticlems/slide2vec?sort=semver&label=docker&logo=docker&color=2496ED)](https://hub.docker.com/r/waticlems/slide2vec)


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
pip install slide2vec
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

   A good starting point is the default configuration file `slide2vec/configs/default.yaml` where parameters are documented.<br>
   We've also added default configuration files for each of the foundation models currently supported:
   - tile-level: `uni`, `uni2`, `virchow`, `virchow2`, `prov-gigapath`, `h-optimus-0`, `h-optimus-1`
   - slide-level: `prov-gigapath`, `titan`, `prism`


3. Kick off distributed feature extraction

    ```shell
    python3 -m slide2vec.main --config-file </path/to/config.yaml>
    ```