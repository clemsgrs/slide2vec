import os
import cv2
import tqdm
import math
import warnings
import numpy as np
import wholeslidedata as wsd
import multiprocessing as mp

from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple

from slide2vec.wsi.utils import find_common_spacings, HasEnoughTissue


# ignore all warnings from wholeslidedata
warnings.filterwarnings("ignore", module="wholeslidedata")

Image.MAX_IMAGE_PIXELS = 933120000


class WholeSlideImage(object):
    def __init__(
        self,
        path: Path,
        mask_path: Optional[Path] = None,
        spacing: Optional[float] = None,
        downsample: int = 64,
        backend: str = "asap",
        tissue_val: int = 1,
        segment: bool = False,
        segment_params: Dict = {
            "sthresh": 8,
            "sthresh_up": 255,
            "mthresh": 7,
            "close": 4,
            "use_otsu": False,
        },
    ):
        """
        Args:
            path (Path): fullpath to WSI file
        """

        self.path = path
        self.name = path.stem.replace(" ", "_")
        self.fmt = path.suffix
        self.wsi = wsd.WholeSlideImage(path, backend=backend)

        self.downsample = downsample
        self.spacing = spacing  # manually set spacing at level 0
        self.spacings = self.get_spacings()
        self.level_dimensions = self.wsi.shapes
        self.level_downsamples = self.get_downsamples()
        self.backend = backend

        self.mask_path = mask_path
        if mask_path is not None:
            tqdm.tqdm.write(f"Loading mask: {mask_path}")
            self.mask = wsd.WholeSlideImage(mask_path, backend=backend)
            self.seg_level = self.load_segmentation(downsample, tissue_val=tissue_val)
        elif segment:
            tqdm.tqdm.write("Segmenting tissue")
            self.seg_level = self.segment_tissue(downsample, **segment_params)

    def get_slide(self, spacing: float):
        return self.wsi.get_slide(spacing=spacing)

    def get_tile(self, x: int, y: int, width: int, height: int, spacing: float):
        return self.wsi.get_patch(x, y, width, height, spacing=spacing, center=False)

    def get_downsamples(self):
        level_downsamples = []
        dim_0 = self.level_dimensions[0]
        for dim in self.level_dimensions:
            level_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(level_downsample)
        return level_downsamples

    def get_spacings(self):
        if self.spacing is None:
            spacings = self.wsi.spacings
        else:
            spacings = [
                self.spacing * s / self.wsi.spacings[0] for s in self.wsi.spacings
            ]
        return spacings

    def get_level_spacing(self, level: int = 0):
        return self.spacings[level]

    def get_best_level_for_spacing(
        self, target_spacing: float, ignore_warning: bool = False
    ):
        spacing = self.get_level_spacing(0)
        target_downsample = target_spacing / spacing
        level, tol, above_tol = self.get_best_level_for_downsample_custom(
            target_downsample, return_tol_status=True
        )
        level_spacing = self.get_level_spacing(level)
        resize_factor = int(round(target_spacing / level_spacing, 0))
        if above_tol and not ignore_warning:
            print(
                f"WARNING! The natural spacing ({resize_factor*self.spacings[level]:.4f}) closest to the target spacing ({target_spacing:.4f}) was more than {tol*100:.1f}% appart ({self.name})."
            )
        return level, resize_factor

    def get_best_level_for_downsample_custom(
        self, downsample, tol: float = 0.1, return_tol_status: bool = False
    ):
        level = int(np.argmin([abs(x - downsample) for x, _ in self.level_downsamples]))
        delta = abs(self.level_downsamples[level][0] / downsample - 1)
        above_tol = delta > tol
        if return_tol_status:
            return level, tol, above_tol
        else:
            return level

    def load_segmentation(
        self,
        downsample: int,
        sthresh_up: int = 255,
        tissue_val: int = 1,
    ):
        # ensure mask and slide have at least one common spacing
        common_spacings = find_common_spacings(
            self.spacings, self.mask.spacings, tolerance=0.1
        )
        assert (
            len(common_spacings) >= 1
        ), f"The provided segmentation mask (spacings={self.mask.spacings}) has no common spacing with the slide (spacings={self.spacings}). A minimum of 1 common spacing is required."

        seg_level = self.get_best_level_for_downsample_custom(downsample)
        seg_spacing = self.get_level_spacing(seg_level)

        # check if this spacing is present in common spacings
        is_in_common_spacings = seg_spacing in [s for s, _ in common_spacings]
        if not is_in_common_spacings:
            # find spacing that is common to slide and mask and that is the closest to seg_spacing
            closest = np.argmin([abs(seg_spacing - s) for s, _ in common_spacings])
            closest_common_spacing = common_spacings[closest][0]
            seg_spacing = closest_common_spacing
            seg_level, _ = self.get_best_level_for_spacing(
                seg_spacing, ignore_warning=True
            )

        m = self.mask.get_slide(spacing=seg_spacing)
        m = m[..., 0]

        m = (m == tissue_val).astype("uint8")
        if np.max(m) <= 1:
            m = m * sthresh_up

        self.binary_mask = m
        return seg_level

    def segment_tissue(
        self,
        downsample: int,
        sthresh: int = 20,
        sthresh_up: int = 255,
        mthresh: int = 7,
        close: int = 0,
        use_otsu: bool = False,
    ):
        """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """

        seg_level = self.get_best_level_for_downsample_custom(downsample)
        seg_spacing = self.get_level_spacing(seg_level)

        img = self.wsi.get_slide(spacing=seg_spacing)
        img = np.array(Image.fromarray(img).convert("RGBA"))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # apply median blurring

        # thresholding
        if use_otsu:
            _, img_thresh = cv2.threshold(
                img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
            )
        else:
            _, img_thresh = cv2.threshold(
                img_med, sthresh, sthresh_up, cv2.THRESH_BINARY
            )

        # morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        self.binary_mask = img_thresh
        return seg_level

    def visualize_mask(
        self,
        contours,
        holes,
        color: Tuple[int] = (0, 255, 0),
        hole_color: Tuple[int] = (0, 0, 255),
        line_thickness: int = 250,
        max_size: Optional[int] = None,
        number_contours: bool = False,
    ):
        vis_level = self.get_best_level_for_downsample_custom(self.downsample)
        level_downsample = self.level_downsamples[vis_level]
        scale = [1 / level_downsample[0], 1 / level_downsample[1]]

        s = self.spacings[vis_level]
        img = self.wsi.get_slide(spacing=s)
        if self.backend == "openslide":
            img = np.ascontiguousarray(img)

        offset = tuple(-(np.array((0, 0)) * scale).astype(int))
        line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
        if contours is not None:
            if not number_contours:
                cv2.drawContours(
                    img,
                    self.scaleContourDim(contours, scale),
                    -1,
                    color,
                    line_thickness,
                    lineType=cv2.LINE_8,
                    offset=offset,
                )

            else:
                # add numbering to each contour
                for idx, cont in enumerate(contours):
                    contour = np.array(self.scaleContourDim(cont, scale))
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / (M["m00"] + 1e-9))
                    cY = int(M["m01"] / (M["m00"] + 1e-9))
                    # draw the contour and put text next to center
                    cv2.drawContours(
                        img,
                        [contour],
                        -1,
                        color,
                        line_thickness,
                        lineType=cv2.LINE_8,
                        offset=offset,
                    )
                    cv2.putText(
                        img,
                        "{}".format(idx),
                        (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        10,
                    )

            for holes in holes:
                cv2.drawContours(
                    img,
                    self.scaleContourDim(holes, scale),
                    -1,
                    hole_color,
                    line_thickness,
                    lineType=cv2.LINE_8,
                )

        img = Image.fromarray(img)

        w, h = img.size
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def get_tile_coordinates(
        self,
        target_spacing: float,
        target_tile_size: int,
        tiling_params: Dict,
        num_workers: int = 1,
    ):
        contours, holes = self.detect_contours(target_spacing, tiling_params)
        (
            running_x_coords,
            running_y_coords,
            tissue_percentages,
            tile_level,
            resize_factor,
        ) = self.process_contours(
            contours,
            holes,
            target_spacing,
            target_tile_size,
            tiling_params["overlap"],
            tiling_params["drop_holes"],
            tiling_params["tissue_thresh"],
            tiling_params["use_padding"],
            num_workers=num_workers,
        )
        tile_coordinates = list(zip(running_x_coords, running_y_coords))
        return (
            contours,
            holes,
            tile_coordinates,
            tissue_percentages,
            tile_level,
            resize_factor,
        )

    def detect_contours(
        self,
        target_spacing: float,
        tiling_params: Dict[str, int],
    ):
        def _filter_contours(contours, hierarchy, tiling_params):
            """
            Filter contours by: area.
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
            all_holes = []

            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                # actual contour
                cont = contours[cont_idx]
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # take contour area (includes holes)
                a = cv2.contourArea(cont)
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                if a == 0:
                    continue
                if a > tiling_params["a_t"]:
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]

            hole_contours = []
            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]
                unfilered_holes = sorted(
                    unfiltered_holes, key=cv2.contourArea, reverse=True
                )
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[: tiling_params["max_n_holes"]]
                filtered_holes = []

                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > tiling_params["a_h"]:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        spacing_level, resize_factor = self.get_best_level_for_spacing(
            target_spacing, ignore_warning=True
        )
        current_scale = self.level_downsamples[spacing_level]
        target_scale = self.level_downsamples[self.seg_level]
        scale = tuple(a / b for a, b in zip(target_scale, current_scale))
        ref_tile_size = tiling_params["ref_tile_size"]
        scaled_ref_tile_area = int(ref_tile_size**2 / (scale[0] * scale[1]))

        _tiling_params = tiling_params.copy()
        _tiling_params["a_t"] = tiling_params["a_t"] * scaled_ref_tile_area
        _tiling_params["a_h"] = tiling_params["a_h"] * scaled_ref_tile_area

        # find and filter contours
        contours, hierarchy = cv2.findContours(
            self.binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        # filtering out artifacts
        foreground_contours, hole_contours = _filter_contours(
            contours, hierarchy, tiling_params
        )

        # scale detected contours to level 0
        contours = self.scaleContourDim(foreground_contours, target_scale)
        holes = self.scaleHolesDim(hole_contours, target_scale)
        return contours, holes

    @staticmethod
    def isInHoles(holes, pt, tile_size):
        for hole in holes:
            if (
                cv2.pointPolygonTest(
                    hole, (pt[0] + tile_size / 2, pt[1] + tile_size / 2), False
                )
                > 0
            ):
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, drop_holes=True, tile_size=256):
        keep_flag, tissue_pct = cont_check_fn(pt)
        if keep_flag:
            if holes is not None and drop_holes:
                return not WholeSlideImage.isInHoles(holes, pt, tile_size), tissue_pct
            else:
                return 1, tissue_pct
        return 0, tissue_pct

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype="int32") for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [
            [np.array(hole * scale, dtype="int32") for hole in holes]
            for holes in contours
        ]

    def process_contours(
        self,
        contours,
        holes,
        spacing: float,
        tile_size: int,
        overlap: float,
        drop_holes: bool,
        tissue_thresh: float,
        use_padding: bool,
        num_workers: int = 1,
    ):
        running_x_coords, running_y_coords = [], []
        running_tissue_pct = []
        tile_level = None
        resize_factor = None

        with tqdm.tqdm(
            contours,
            desc="Processing tissue blobs",
            unit=" contour",
            total=len(contours),
            leave=False,
        ) as t:
            for i, cont in enumerate(t):
                (
                    x_coords,
                    y_coords,
                    tissue_pct,
                    cont_tile_level,
                    cont_resize_factor,
                ) = self.process_contour(
                    cont,
                    holes[i],
                    spacing,
                    tile_size,
                    overlap,
                    drop_holes,
                    tissue_thresh,
                    use_padding,
                    num_workers=num_workers,
                )
                if len(x_coords) > 0:
                    if tile_level is not None:
                        assert (
                            tile_level == cont_tile_level
                        ), "tile level should be the same for all contours"
                    tile_level = cont_tile_level
                    resize_factor = cont_resize_factor
                    running_x_coords.extend(x_coords)
                    running_y_coords.extend(y_coords)
                    running_tissue_pct.extend(tissue_pct)

        return (
            running_x_coords,
            running_y_coords,
            running_tissue_pct,
            tile_level,
            resize_factor,
        )

    def process_contour(
        self,
        contour,
        contour_holes,
        spacing: float,
        tile_size: int,
        overlap: float,
        drop_holes: bool,
        tissue_thresh: float,
        use_padding: bool,
        spacing_tol: float = 0.15,
        num_workers: int = 1,
    ):
        tile_level, _ = self.get_best_level_for_spacing(spacing, ignore_warning=True)

        tile_spacing = self.get_level_spacing(tile_level)
        resize_factor = int(round(spacing / tile_spacing, 0))

        if abs(resize_factor * tile_spacing / spacing - 1) > spacing_tol:
            raise ValueError(
                f"ERROR: The natural spacing ({resize_factor*tile_spacing:.4f}) closest to the target spacing ({spacing:.4f}) was more than {spacing_tol*100}% apart."
            )

        tile_size_resized = tile_size * resize_factor
        step_size = int(tile_size_resized * (1.0 - overlap))

        if contour is not None:
            start_x, start_y, w, h = cv2.boundingRect(contour)
        else:
            start_x, start_y, w, h = (
                0,
                0,
                self.level_dimensions[tile_level][0],
                self.level_dimensions[tile_level][1],
            )

        # 256x256 tiles at 1mpp are equivalent to 512x512 tiles at 0.5mpp
        # ref_tile_size capture the tile size at level 0
        # assumes self.level_downsamples[0] is always (1, 1)
        tile_downsample = (
            int(self.level_downsamples[tile_level][0]),
            int(self.level_downsamples[tile_level][1]),
        )
        ref_tile_size = (
            tile_size_resized * tile_downsample[0],
            tile_size_resized * tile_downsample[1],
        )

        img_w, img_h = self.level_dimensions[0]
        if use_padding:
            stop_y = int(start_y + h)
            stop_x = int(start_x + w)
        else:
            stop_y = min(start_y + h, img_h - ref_tile_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_tile_size[0] + 1)

        scale = self.level_downsamples[self.seg_level]
        cont = self.scaleContourDim([contour], (1.0 / scale[0], 1.0 / scale[1]))[0]
        cont_check_fn = HasEnoughTissue(
            contour=cont,
            contour_holes=contour_holes,
            tissue_mask=self.binary_mask,
            tile_size=ref_tile_size[0],
            scale=scale,
            pct=tissue_thresh,
        )

        # input step_size is defined w.r.t to input spacing
        # given contours are defined w.r.t level 0, step_size (potentially) needs to be upsampled
        ref_step_size_x = int(step_size * tile_downsample[0])
        ref_step_size_y = int(step_size * tile_downsample[1])

        # x & y values are defined w.r.t level 0
        x_range = np.arange(start_x, stop_x, step=ref_step_size_x)
        y_range = np.arange(start_y, stop_y, step=ref_step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
        coord_candidates = np.array(
            [x_coords.flatten(), y_coords.flatten()]
        ).transpose()

        if num_workers > 1:
            num_workers = min(mp.cpu_count(), num_workers)
            if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
                num_workers = min(
                    num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
                )

            pool = mp.Pool(num_workers)

            iterable = [
                (coord, contour_holes, ref_tile_size[0], cont_check_fn, drop_holes)
                for coord in coord_candidates
            ]
            results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
            pool.close()
            filtered_coordinates = np.array(
                [result[0] for result in results if result[0] is not None]
            )
            filtered_tissue_percentages = [
                result[1] for result in results if result[0] is not None
            ]
        else:
            coordinates = []
            tissue_percentages = []
            for coord in coord_candidates:
                c, pct = self.process_coord_candidate(
                    coord, contour_holes, ref_tile_size[0], cont_check_fn, drop_holes
                )
                coordinates.append(c)
                tissue_percentages.append(pct)
            filtered_coordinates = np.array(
                [coordinate for coordinate in coordinates if coordinate is not None]
            )
            filtered_tissue_percentages = [
                tissue_percentages[i]
                for i, coordinate in enumerate(coordinates)
                if coordinate is not None
            ]

        ntile = len(filtered_coordinates)

        if ntile > 0:
            x_coords = list(filtered_coordinates[:, 0])
            y_coords = list(filtered_coordinates[:, 1])
            return (
                x_coords,
                y_coords,
                filtered_tissue_percentages,
                tile_level,
                resize_factor,
            )

        else:
            return [], [], [], None, None

    @staticmethod
    def process_coord_candidate(
        coord, contour_holes, tile_size, cont_check_fn, drop_holes
    ):
        keep_flag, tissue_pct = WholeSlideImage.isInContours(
            cont_check_fn, coord, contour_holes, drop_holes, tile_size
        )
        if keep_flag:
            return coord, tissue_pct
        else:
            return None, tissue_pct
