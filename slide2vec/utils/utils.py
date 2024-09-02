import pandas as pd


def load_csv(cfg):
    df  = pd.read_csv(cfg.csv)
    wsi_paths = df.wsi_path.values.tolist()
    if "mask_path" in df.columns:
        mask_paths = df.mask_path.values.tolist()
    else:
        mask_paths = [None for _ in wsi_paths]
    return wsi_paths, mask_paths