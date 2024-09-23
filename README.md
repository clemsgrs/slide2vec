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

## Extract UNI features

```shell
torchrun --nproc_per_node=gpu slide2vec/main.py \
    --config-file slide2vec/configs/uni.yaml \
```