# slide2vec

:warning: Make sure the `slide2vec` package is included in the Python module search path:

```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/slide2vec"
```

## Extract UNI features

```shell
python3 -m torch.distributed.run --nproc_per_node=gpu slide2vec/main.py \
    --config-file slide2vec/configs/uni.yaml \
```