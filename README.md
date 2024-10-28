# slide2vec

:warning: Make sure the `slide2vec` package is included in your Python module search path:

```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/slide2vec"
```

## Summary of pathology foundation models

| **Architecture** | **Foundation Model** | **Parameters** |
|:----------------:|:--------------------:|:--------------:|
|      ViT-B/8     |         Kaiko        |       86M      |
|     ViT-B/16     |        Phikon        |       86M      |
|     ViT-L/16     |          UNI         |      307M      |
|     ViT-L/14     |         Kaiko        |      307M      |
|     ViT-H/14     |        Virchow       |      632M      |
|     ViT-H/14     |       Virchow2       |      632M      |
|     ViT-G/14     |       GigaPath       |      1.8B      |
|     ViT-G/14     |       H-optimus      |      1.8B      |

## Extract tile-level features

1. Create a `.csv` file with slide and tissue mask paths

    ```csv
    wsi_path,mask_path
    /path/to/slide1.tif,/path/to/mask1.tif
    /path/to/slide2.tif,/path/to/mask2.tif
    ...
    ```

2. Create a configuration file under `slide2vec/configs`

    A good starting point is the default configuration file `slide2vec/configs/default.yaml` where parameters are documented.

3. Kick off distributed feature extraction

    ```shell
    torchrun --nproc_per_node=gpu slide2vec/main.py --config-file slide2vec/configs/uni.yaml
    ```