"""Single source of truth for the output-consistency test and fixture generator."""

TILING_PARAMS = {
    "requested_spacing_um": 0.5,
    "tolerance": 0.07,
    "requested_tile_size_px": 224,
    "overlap": 0.0,
}

TILING_MASKS = {
    "output_mode": "per_annotation",
    "pixel_mapping": {"background": 0, "tissue": 1},
    "colors": {"background": None, "tissue": [157, 219, 129]},
    "min_coverage": {"background": None, "tissue": 0.1},
}

TILING_SEG_PARAMS = {
    "downsample": 64,
    "sthresh": 8,
    "sthresh_up": 255,
    "mthresh": 7,
    "close": 4,
    "method": "hsv",
}

TILING_FILTER_PARAMS = {
    "ref_tile_size": 224,
    "a_t": 4,
    "a_h": 2,
    "filter_white": False,
    "filter_black": False,
    "white_threshold": 220,
    "black_threshold": 25,
    "fraction_threshold": 0.9,
}

TILING_PREVIEW = {
    "save_mask_preview": False,
    "save_tiling_preview": False,
    "downsample": 32,
    "tissue_contour_color": (157, 219, 129),
    "mask_overlay_alpha": 0.5,
}

MODEL_PARAMS = {
    "name": "prism",
    "batch_size": 8,
    "save_tile_embeddings": True,
    "save_slide_embeddings": False,
    "save_latents": False,
}

SPEED_PARAMS = {
    "precision": "fp16",
    "num_dataloader_workers": 0,
}
