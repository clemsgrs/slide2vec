# slide2vec

:warning: Make sure to run the following:

```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/slide2vec"
export HF_TOKEN=<your-huggingface-api-token>
```

## Extract features

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
    python3 slide2vec/main.py --config-file </path/to/config.yaml>
    ```