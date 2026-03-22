import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cv2
import os
import time
import glob
import re
import os, glob
import numpy as np
import cv2
import xarray as xr

def frazil_spectrum(path_v, video_file, start_time, thresh=40, retrieval_mode="RETR_TREE"):
    """
    Estimate frazil crystal geometry from Icefin images scaled with the brick.

    Legacy behavior preserved:
      - cv2.blur((3,3))
      - binary threshold
      - findContours default RETR_TREE
      - record major-axis for EVERY contour using j-indexing
      - same fractional area logic and NetCDF layout

    Adds:
      - first pass to size max_crystal_count from raw contour counts
      - robust checks & explicit errors (no silent None returns)
    """
    # Imaging parameters 
    scale = 2.5 / 1920.0  # metres per pixel

    # Paths 
    image_folder = os.path.join(path_v, video_file[:-10] + "_frames_" + str(start_time))
    background_subtracted_path = os.path.join(image_folder, "background_subtracted")
    netcdf_path = os.path.join(path_v, "netcdf_frazil_data")
    os.makedirs(netcdf_path, exist_ok=True)
    binary_path = os.path.join(background_subtracted_path, "binary")

    out_nc = os.path.join(
        netcdf_path,
        f"{video_file[:-10]}_thresh_{thresh}_mins_{int(start_time)}.nc"
    )
    # Map retrieval mode
    mode = cv2.RETR_TREE if str(retrieval_mode).upper() == "RETR_TREE" else cv2.RETR_EXTERNAL
    # If dataset exists, return it
    if os.path.exists(out_nc):
        ds = xr.open_dataset(out_nc, engine="netcdf4")
        if ds is None:
            raise RuntimeError(f"xr.open_dataset returned None for: {out_nc}")
        return ds
    else:
        # Collect image files (sorting by ctime)
        image_files = sorted(
            glob.glob(os.path.join(background_subtracted_path, "*foreground_clahe_masked.jpg"))
        )
        image_files = sorted(image_files, key=frame_index_key)
        if len(image_files) == 0:
            raise FileNotFoundError(
                f"No images found at: {background_subtracted_path} "
                f"with pattern '*foreground_clahe_masked.jpg'"
            )
        # ---------- FIRST PASS: raw contour counts using chosen mode ----------
        raw_counts = []
        rel_areas = []        # fraction in [0, 1]
        for processed_image_path in image_files:
            img = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Could not read image: {processed_image_path}")
            blurred = cv2.blur(img, (3, 3), 0)  # legacy
            _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
            
            # Relative pixel area (fraction in [0, 1])
            rel_area = cv2.countNonZero(binary) / binary.size
            rel_areas.append(rel_area)
            contours, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
            raw_counts.append(len(contours))

        if len(raw_counts) == 0:
            raise RuntimeError("First pass found zero frames with contours; check inputs/threshold.")
        # If you want relative area in percentage:
        rel_area_pct = 100.0 * np.array(rel_areas)
        max_raw = max(raw_counts)
        # Small safety margin to avoid off-by-one spikes
        max_crystal_count = int(max_raw * 1.05) + 1
        if max_crystal_count <= 0:
            max_crystal_count = 1  # ensure positive width

        # ---------- Allocate legacy arrays ----------
        total_rectangle_area = []
        length_major_axis = np.empty((len(image_files), max_crystal_count), dtype=float)
        length_major_axis[:] = np.nan
        contour_count = []  # QA only

        # ---------- Main pass (j-indexing, same drawing) ----------
        im_count = 1
        for processed_image_path in image_files:
            processed_image = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
            if processed_image is None:
                raise IOError(f"Could not read image: {processed_image_path}")

            blurred_image = cv2.blur(processed_image, (3, 3), 0)
            _, binary_image = cv2.threshold(blurred_image, thresh, 255, cv2.THRESH_BINARY)
            binary_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

            contours, hierarchy = cv2.findContours(binary_image, mode, cv2.CHAIN_APPROX_SIMPLE)
            contour_count.append(len(contours))

    #        rectangle_pixel_area = []
            j = 0
            for contour in contours:
                # Hard guard: never overflow second dimension
                if j >= max_crystal_count:
                    # Optional: log truncation
                    # print(f"[WARN] Frame {im_count-1}: {len(contours)} contours; truncating at {max_crystal_count}.")
                    break

                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                # Calculate the frazil ice diameter by picking the major axis and adding a pixel, because the algorithm picked pixel centres
                length_major_axis[im_count-1, j] = np.max(rect[1][:]) * scale  + 1 * scale # unit metre

                if width > 0 and height > 0:
                    # Legacy elongated check
                    if (width / height) > 1 or (height / width) > 1:
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        cv2.drawContours(binary_rgb, [box], 0, (255, 0, 0), 2)
                j += 1

            # Save images only once
            if not os.path.exists(os.path.join(binary_path, "with_contours")):
                os.makedirs(binary_path, exist_ok=True)
                os.makedirs(os.path.join(binary_path, "with_contours"), exist_ok=True)
            if im_count==1:
                cv2.imwrite(os.path.join(binary_path, "with_contours", f"{im_count}.jpg"), binary_rgb)
                cv2.imwrite(os.path.join(binary_path, f"{im_count}.jpg"), binary_image)
            else:
                os.remove(processed_image_path)

            im_count += 1

        # ---------- Build dataset  ----------
        frame_time = np.arange(len(rel_area_pct)) * (1/30.0)
        frame_number = np.arange(len(rel_area_pct))
        fractional_frazil_area = np.array(rel_area_pct) / (1080 * 1920)

        ds = xr.Dataset(
            {
                "fractional_frazil_area": (["frame_number"], fractional_frazil_area),
                "length_major_axis": (["frame_number", "crystal_number"], length_major_axis),
                "rel_area": (["frame_number"], rel_area_pct),
            },
            coords={
                "frame_number": frame_number,
                "frame_time": ("frame_number", frame_time),
                "crystal_number": np.arange(max_crystal_count),
            },
            attrs={
                "pixel_to_meter_scale": scale,
                "threshold": int(thresh),
                "retrieval_mode": "RETR_TREE" if mode == cv2.RETR_TREE else "RETR_EXTERNAL",
                "sorted_by": "ctime",
                "max_raw_contours": int(max_raw),
                "allocated_crystal_count": int(max_crystal_count),
            }
        )

        # Write and return
        ds.to_netcdf(out_nc, engine="netcdf4")
    return ds

def frame_index_key(path: str) -> int:
    name = os.path.basename(path)
    # Extract the last number in the filename; adjust the regex to your pattern if needed
    m = re.findall(r'(\d+)', name)
    return int(m[-1]) if m else -1  # put “numberless” files first (or raise)


def analyze_IcefinCam(ds_sizes, tau=1/30, max_range=60.0, bin_width=1e-3):
    """Estimate frazil-ice crystal geometry and concentration from Icefin images.

    Args:
        ds_sizes (xarray.Dataset): Frail-ice diameters (major axis lengths, in meters).
        tau (float): Thickness-to-diameter aspect ratio of frazil ice (Frazer et al. 2020).
        max_range (float): Maximum diameter of interest in millimeters.
        bin_width (float): Bin width for the diameter distribution (meters).

    Returns:
        xarray.Dataset: Frazil-ice concentration and size statistics.
                        Concentrations are in m^3 m^{-3} (volume fraction).
    """

    # Convert max_range from mm to meters
    max_range *= 1e-3

    # Imaging geometry
    w = 2.5     # m
    h = 1.4     # m
    l = 1.5     # m
    w_mask = 0.728
    h_mask = 0.793

    V_frame = (w * h * l)/3 - (w_mask * h_mask * l)/3  # m^3

    # Extract diameters (meters)
    diameters = ds_sizes.length_major_axis.data

    conc_per_frame = []
    for i in range(diameters.shape[0]):
        d = diameters[i, :]
        d = d[~np.isnan(d)]

        if d.size > 0:
            vols = (np.pi / 4) * tau * d**3  # crystal volumes (m^3)
            conc = np.sum(vols) / V_frame   # m^3/m^3
        else:
            conc = np.nan

        conc_per_frame.append(conc)

    conc_per_frame = np.array(conc_per_frame)

    # Flatten for overall size distribution
    diam_all = diameters.flatten()
    diam_all = diam_all[~np.isnan(diam_all)]
    vols_all = (np.pi / 4) * tau * diam_all**3  # m^3

    # Define bins in meters
    bins = np.arange(0, max_range + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    counts, _ = np.histogram(diam_all, bins=bins)
    vol_dist, _ = np.histogram(diam_all, bins=bins, weights=vols_all)

    normalized_counts = counts / counts.sum()

    number_density = counts.sum() / (V_frame * len(ds_sizes.frame_number))

    frazil_stats = xr.Dataset(
        data_vars={
            "meanFIC": (["depth"], [np.nanmean(conc_per_frame)]),   # m^3/m^3
            "medianFIC": (["depth"], [np.nanmedian(conc_per_frame)]),
            "stdFIC": (["depth"], [np.nanstd(conc_per_frame)]),
            "meanD": (["depth"], [np.nanmean(diam_all)]),
            "medianD": (["depth"], [np.nanmedian(diam_all)]),
            "stdD": (["depth"], [np.nanstd(diam_all)]),
            "N_obs": (["depth"], [len(ds_sizes.frame_number)]),
            "normalized_numbers": (["diameter"], normalized_counts),
            "number_of_crystals": (["depth"], [counts.sum()]),
            "number_density": (["depth"], [number_density]),
            "bulk_concentration": (["depth"], [np.sum(vols_all) / (len(ds_sizes.frame_number) * V_frame)]),
            "volume_distribution": (["diameter"], vol_dist / (len(ds_sizes.frame_number) * V_frame)),
            "rel_area": ("depth", [np.nanmean(ds_sizes.rel_area)]),
        },
        coords={
            "depth": np.array([np.nan]),
            "diameter": bin_centers,
        },
    )
    return frazil_stats
