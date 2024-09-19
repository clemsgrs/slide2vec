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
        print(f"self.spacings: {self.spacings}")
        self.level_dimensions = self.wsi.shapes
        self.level_downsamples = self.get_downsamples()
        print(f"self.level_downsamples: {self.level_downsamples}")
        self.backend = backend

        self.mask_path = mask_path
        if mask_path is not None:
            tqdm.tqdm.write(f"Loading mask: {mask_path}")
            self.mask = wsd.WholeSlideImage(mask_path, backend=backend)
            self.seg_level = self.load_segmentation(downsample, tissue_val=tissue_val)
        elif segment:
            tqdm.tqdm.write("Segmenting tissue")
            self.seg_level = self.segment_tissue(downsample)

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

    def get_patch_coordinates(
        self,
        target_spacing,
        target_patch_size,
        patching_params: Dict[str, int] = {
            "overlap": 0.0,
            "drop_holes": False,
            "tissue_thresh": 0.01,
            "use_padding": True,
        },
        filter_params: Dict[str, int] = {
            "ref_patch_size": 16,
            "a_t": 4,
            "a_h": 2,
            "max_n_holes": 8,
        },
        num_workers: int = 1,
    ):
        contours, holes = self.detect_contours(target_spacing, filter_params)
        (
            running_x_coords,
            running_y_coords,
            tissue_percentages,
            patch_level,
            resize_factor,
        ) = self.process_contours(
            contours,
            holes,
            spacing=target_spacing,
            patch_size=target_patch_size,
            overlap=patching_params["overlap"],
            drop_holes=patching_params["drop_holes"],
            tissue_thresh=patching_params["tissue_thresh"],
            use_padding=patching_params["use_padding"],
            num_workers=num_workers,
        )
        patch_coordinates = list(zip(running_x_coords, running_y_coords))
        return (
            contours,
            holes,
            patch_coordinates,
            tissue_percentages,
            patch_level,
            resize_factor,
        )

    def detect_contours(
        self,
        target_spacing: float,
        filter_params: Dict[str, int],
    ):
        def _filter_contours(contours, hierarchy, filter_params):
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
                if a > filter_params["a_t"]:
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
                unfilered_holes = unfilered_holes[: filter_params["max_n_holes"]]
                filtered_holes = []

                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params["a_h"]:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        spacing_level, resize_factor = self.get_best_level_for_spacing(
            target_spacing, ignore_warning=True
        )
        current_scale = self.level_downsamples[spacing_level]
        target_scale = self.level_downsamples[self.seg_level]
        scale = tuple(a / b for a, b in zip(target_scale, current_scale))
        ref_patch_size = filter_params["ref_patch_size"]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))

        filter_params = filter_params.copy()
        filter_params["a_t"] = filter_params["a_t"] * scaled_ref_patch_area
        filter_params["a_h"] = filter_params["a_h"] * scaled_ref_patch_area

        # find and filter contours
        contours, hierarchy = cv2.findContours(
            self.binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        print(f"len(contours): {len(contours)}")
        if filter_params:
            # necessary for filtering out artifacts
            foreground_contours, hole_contours = _filter_contours(
                contours, hierarchy, filter_params
            )

        # scale detected contours to level 0
        print(f"len(foreground_contours): {len(foreground_contours)}")
        contours = self.scaleContourDim(foreground_contours, target_scale)
        holes = self.scaleHolesDim(hole_contours, target_scale)
        return contours, holes

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if (
                cv2.pointPolygonTest(
                    hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False
                )
                > 0
            ):
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, drop_holes=True, patch_size=256):
        keep_flag, tissue_pct = cont_check_fn(pt)
        if keep_flag:
            if holes is not None and drop_holes:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size), tissue_pct
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
        spacing: float = 0.5,
        patch_size: int = 256,
        overlap: float = 0.0,
        drop_holes: bool = False,
        tissue_thresh: float = 0.1,
        use_padding: bool = True,
        num_workers: int = 1,
    ):
        running_x_coords, running_y_coords = [], []
        running_tissue_pct = []
        patch_level = None
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
                    cont_patch_level,
                    cont_resize_factor,
                ) = self.process_contour(
                    cont,
                    holes[i],
                    spacing,
                    patch_size,
                    overlap,
                    drop_holes,
                    tissue_thresh,
                    use_padding,
                    num_workers=num_workers,
                )
                if len(x_coords) > 0:
                    if patch_level is not None:
                        assert (
                            patch_level == cont_patch_level
                        ), "Patch level should be the same for all contours"
                    patch_level = cont_patch_level
                    resize_factor = cont_resize_factor
                    running_x_coords.extend(x_coords)
                    running_y_coords.extend(y_coords)
                    running_tissue_pct.extend(tissue_pct)

        return (
            running_x_coords,
            running_y_coords,
            running_tissue_pct,
            patch_level,
            resize_factor,
        )

    def process_contour(
        self,
        contour,
        contour_holes,
        spacing: float,
        patch_size: int = 256,
        overlap: float = 0.0,
        drop_holes: bool = True,
        tissue_thresh: float = 0.01,
        use_padding: bool = True,
        num_workers: int = 1,
    ):
        patch_level, resize_factor = self.get_best_level_for_spacing(
            spacing, ignore_warning=True
        )

        patch_spacing = self.get_level_spacing(patch_level)
        resize_factor = int(round(spacing / patch_spacing, 0))
        patch_size_resized = patch_size * resize_factor
        step_size = int(patch_size_resized * (1.0 - overlap))

        if contour is not None:
            start_x, start_y, w, h = cv2.boundingRect(contour)
        else:
            start_x, start_y, w, h = (
                0,
                0,
                self.level_dimensions[patch_level][0],
                self.level_dimensions[patch_level][1],
            )

        # 256x256 patches at 1mpp are equivalent to 512x512 patches at 0.5mpp
        # ref_patch_size capture the patch size at level 0
        # assumes self.level_downsamples[0] is always (1, 1)
        patch_downsample = (
            int(self.level_downsamples[patch_level][0]),
            int(self.level_downsamples[patch_level][1]),
        )
        ref_patch_size = (
            patch_size_resized * patch_downsample[0],
            patch_size_resized * patch_downsample[1],
        )

        img_w, img_h = self.level_dimensions[0]
        if use_padding:
            stop_y = int(start_y + h)
            stop_x = int(start_x + w)
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

        scale = self.level_downsamples[self.seg_level]
        cont = self.scaleContourDim([contour], (1.0 / scale[0], 1.0 / scale[1]))[0]
        cont_check_fn = HasEnoughTissue(
            contour=cont,
            contour_holes=contour_holes,
            tissue_mask=self.binary_mask,
            patch_size=ref_patch_size[0],
            scale=scale,
            pct=tissue_thresh,
        )

        # input step_size is defined w.r.t to input spacing
        # given contours are defined w.r.t level 0, step_size (potentially) needs to be upsampled
        ref_step_size_x = int(step_size * patch_downsample[0])
        ref_step_size_y = int(step_size * patch_downsample[1])

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
                (coord, contour_holes, ref_patch_size[0], cont_check_fn, drop_holes)
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
                    coord, contour_holes, ref_patch_size[0], cont_check_fn, drop_holes
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

        npatch = len(filtered_coordinates)

        if npatch > 0:
            x_coords = list(filtered_coordinates[:, 0])
            y_coords = list(filtered_coordinates[:, 1])
            return (
                x_coords,
                y_coords,
                filtered_tissue_percentages,
                patch_level,
                resize_factor,
            )

        else:
            return [], [], [], None, None

    @staticmethod
    def process_coord_candidate(
        coord, contour_holes, patch_size, cont_check_fn, drop_holes
    ):
        keep_flag, tissue_pct = WholeSlideImage.isInContours(
            cont_check_fn, coord, contour_holes, drop_holes, patch_size
        )
        if keep_flag:
            return coord, tissue_pct
        else:
            return None, tissue_pct
