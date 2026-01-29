#!/usr/bin/env python3
"""
Body-map pipeline with:
- fixed analysis mask from interior_mask.png (white=include, black=exclude)
- extraction of grey drawing pixels (semi-transparent black alpha≈64)
- per-emotion ink metrics per participant
- cumulative emotion density maps (proportion inked)
- slider↔emotion ink linkage
- NEW: combined cumulative plot with black background, white text, white uncoloured body,
       coloured intensity increasing with overlap, tight spacing, legend.
- NEW: aware↔emotion overlap metrics per participant and slider linkage
- NEW: slider-weighted aware summary images per emotion
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from scipy import stats


# ------------------------------ Constants ------------------------------

ID_PUBLIC_COL = "Participant Public ID"
ID_PRIVATE_COL = "Participant Private ID"

SLIDER_EMOTIONS = {
    "Happy": "Slider object-2 Value",
    "Sad": "Slider object-3 Value",
    "Anger": "Slider object-4 Value",
    "Fear": "Slider object-5 Value",
    "Disgust": "Slider object-6 Value",
}

EMOTIONS = ["Happy", "Sad", "Anger", "Fear", "Disgust", "Aware"]
EMOTIONS_NO_AWARE = ["Happy", "Sad", "Anger", "Fear", "Disgust"]


# ------------------------------ File system helpers ------------------------------

def _list_images(image_dir: Path) -> List[Path]:
    return sorted(image_dir.glob("*.png"))


def _find_questionnaire(root: Path) -> Path:
    candidates = sorted([p for p in root.iterdir() if p.is_file()])
    for p in candidates:
        if re.search(r"questionnaire", p.name, flags=re.IGNORECASE):
            return p
    for p in candidates:
        if p.suffix.lower() in (".tsv", ".csv"):
            return p
    raise FileNotFoundError(f"No questionnaire-like file found in {root}")


def _read_questionnaire(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, sep="\t")
    except Exception:
        df = pd.read_csv(path, dtype=str)

    if ID_PRIVATE_COL not in df.columns:
        raise KeyError(f"Questionnaire missing required column: {ID_PRIVATE_COL}")
    return df


# ------------------------------ Mask handling ------------------------------

def load_analysis_mask(mask_path: Union[str, os.PathLike, Path]) -> np.ndarray:
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Analysis mask not found: {mask_path}")

    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Could not read mask image: {mask_path}")

    if m.ndim == 3:
        if m.shape[2] == 4:
            b, g, r, a = cv2.split(m)
            m_gray = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)
            include = (m_gray >= 128) & (a > 0)
        else:
            m_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            include = m_gray >= 128
    else:
        include = m >= 128

    include = np.ascontiguousarray(include.astype(bool))
    if not include.any():
        raise ValueError("Analysis mask contains no white/included pixels.")
    return include


def _resize_to_mask(arr: np.ndarray, target_shape_hw: Tuple[int, int]) -> np.ndarray:
    H, W = target_shape_hw
    if arr.ndim == 2:
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
    if arr.ndim == 3:
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
    raise ValueError("Expected 2D or 3D array for resizing.")


# ------------------------------ Image parsing and mapping ------------------------------

def _build_image_table(image_paths: Iterable[Path], questionnaire_path: Path) -> pd.DataFrame:
    qdf = _read_questionnaire(questionnaire_path)

    keep_cols = [ID_PRIVATE_COL] + list(SLIDER_EMOTIONS.values())
    keep_cols = [c for c in keep_cols if c in qdf.columns]
    qdf = qdf[keep_cols].copy()

    img_paths = list(image_paths)

    def _paths_for_pid(pid: str) -> List[Path]:
        pid_str = str(pid)
        return [p for p in img_paths if pid_str in p.name]

    qdf["image_paths"] = qdf[ID_PRIVATE_COL].apply(_paths_for_pid)

    for emotion in EMOTIONS:
        qdf[f"{emotion}_file"] = qdf["image_paths"].apply(
            lambda paths: next((p for p in paths if emotion.lower() in p.name.lower()), pd.NA)
        )

    return qdf


# ------------------------------ Grey extraction within fixed analysis mask ------------------------------

def extract_grey_within_analysis_mask(
    img_in: Union[str, os.PathLike, Image.Image, np.ndarray],
    analysis_mask: np.ndarray,
    *,
    grey_rgb_tol: int = 10,
    grey_alpha_target: int = 64,
    grey_alpha_tol: int = 6,
) -> np.ndarray:
    if isinstance(img_in, (str, os.PathLike)):
        img = cv2.imread(str(img_in), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {img_in}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

        if img.shape[2] == 4:
            b, g, r, a = cv2.split(img)
            rgba = cv2.merge([r, g, b, a])
        elif img.shape[2] == 3:
            b, g, r = cv2.split(img)
            a = np.full_like(b, 255, dtype=np.uint8)
            rgba = cv2.merge([r, g, b, a])
        else:
            raise ValueError("Unexpected channel count in image.")
    elif isinstance(img_in, Image.Image):
        rgba = np.array(img_in.convert("RGBA"), dtype=np.uint8)
    elif isinstance(img_in, np.ndarray):
        arr = img_in
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("Expected HxWx{3,4} array.")
        if arr.shape[2] == 3:
            a = np.full(arr.shape[:2], 255, dtype=np.uint8)
            rgba = np.dstack([arr, a]).astype(np.uint8)
        else:
            rgba = arr.astype(np.uint8)
    else:
        raise TypeError("img_in must be a path, PIL.Image, or numpy array.")

    Hm, Wm = analysis_mask.shape
    if rgba.shape[0] != Hm or rgba.shape[1] != Wm:
        rgba = _resize_to_mask(rgba, (Hm, Wm)).astype(np.uint8)

    R, G, B, A = [rgba[..., i] for i in range(4)]

    grey_rgb = (R <= grey_rgb_tol) & (G <= grey_rgb_tol) & (B <= grey_rgb_tol)
    a_lo = max(0, grey_alpha_target - grey_alpha_tol)
    a_hi = min(255, grey_alpha_target + grey_alpha_tol)
    grey_alpha_sel = (A >= a_lo) & (A <= a_hi)

    grey_mask = grey_rgb & grey_alpha_sel
    grey_mask_in = grey_mask & analysis_mask
    return grey_mask_in


# ------------------------------ Metrics and averaging ------------------------------

def ink_metrics_for_image(img_path: Path, analysis_mask: np.ndarray) -> Dict[str, float]:
    grey_in = extract_grey_within_analysis_mask(img_path, analysis_mask)
    ink_count = float(grey_in.sum())
    mask_area = float(analysis_mask.sum())
    ink_frac = float(ink_count / mask_area) if mask_area > 0 else 0.0
    return {"ink_count": ink_count, "ink_frac": ink_frac}


def _resolve_path(v: object, image_dir: Path) -> Optional[Path]:
    if v is None or pd.isna(v):
        return None
    p = Path(v)
    if not p.is_absolute():
        p = image_dir / p
    if p.exists():
        return p
    p2 = Path(str(v))
    if p2.exists():
        return p2
    return None


def load_binary_ink_mask_for_participant(
    wide_row: pd.Series,
    emotion: str,
    *,
    image_dir: Path,
    analysis_mask: np.ndarray,
) -> Optional[np.ndarray]:
    p = _resolve_path(wide_row.get(f"{emotion}_file"), image_dir)
    if p is None:
        return None
    try:
        return extract_grey_within_analysis_mask(p, analysis_mask)
    except Exception as e:
        warnings.warn(f"Failed to load ink mask for {emotion} at {p}: {e}")
        return None


def cumulative_density_from_paths(
    paths: List[Path],
    analysis_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Returns density in [0,1] where 0 means nobody coloured, 1 means everyone coloured.
    """
    if not paths:
        raise ValueError("No images provided.")

    H, W = analysis_mask.shape
    stack = np.zeros((len(paths), H, W), dtype=np.float32)

    valid = 0
    for i, p in enumerate(paths):
        if not p.exists():
            continue
        try:
            g = extract_grey_within_analysis_mask(p, analysis_mask).astype(np.float32)
            stack[i] = g
            valid += 1
        except Exception:
            continue

    if valid == 0:
        raise ValueError("All images failed to process.")
    density = stack.mean(axis=0)  # [0,1]
    density[~analysis_mask] = np.nan
    return density, valid


# ------------------------------ Visual styling helpers ------------------------------

def _white_to_colour_cmap(rgb: Tuple[float, float, float], name: str) -> LinearSegmentedColormap:
    """
    Colormap where 0 -> white, 1 -> chosen colour (not black).
    Outside-body will be rendered via 'bad' colour (set separately).
    """
    cmap = LinearSegmentedColormap.from_list(name, [(1, 1, 1), rgb], N=256)
    return cmap


def _emotion_cmap(emotion: str) -> LinearSegmentedColormap:
    # Deliberately saturated but not neon; high values are “strong colour” (perceived darker vs white).
    rgb_by_emotion = {
        "Happy": (0.90, 0.60, 0.05),    # orange
        "Sad": (0.10, 0.35, 0.80),      # blue
        "Anger": (0.80, 0.10, 0.10),    # red
        "Fear": (0.55, 0.20, 0.75),     # purple
        "Disgust": (0.10, 0.60, 0.25),  # green
    }
    rgb = rgb_by_emotion.get(emotion, (0.4, 0.7, 0.9))
    cmap = _white_to_colour_cmap(rgb, f"{emotion}_white_to_colour")
    cmap.set_bad(color=(0, 0, 0, 1))  # outside mask -> black
    return cmap


# ------------------------------ NEW: improved combined plot ------------------------------

def plot_emotion_cumulative_grid_styled(
    wide_df: pd.DataFrame,
    analysis_mask: np.ndarray,
    *,
    image_dir: Union[str, os.PathLike, Path],
    out_path: Union[str, os.PathLike, Path],
    emotions: Optional[List[str]] = None,
    dpi: int = 400,
    panel_height_in: float = 3.0,
    add_legend: bool = True,
) -> None:
    """
    Combined 1×K cumulative maps with:
    - background black,
    - inside-body uncoloured = white,
    - increasing overlap = more intense colour,
    - tight spacing (“bodies closer together”),
    - white text,
    - colourbar legend describing intensity meaning.

    Normalisation is inherently N-invariant because we plot proportion inked in [0,1].
    """
    if emotions is None:
        emotions = EMOTIONS_NO_AWARE

    image_dir = Path(image_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    K = len(emotions)
    # Make panels close: small width per panel and minimal wspace.
    fig_width = panel_height_in * K * 0.75

    fig, axes = plt.subplots(1, K, figsize=(fig_width, panel_height_in), dpi=dpi)
    if K == 1:
        axes = [axes]

    fig.patch.set_facecolor("black")

    norm = Normalize(vmin=0.0, vmax=1.0)

    for ax, emotion in zip(axes, emotions):
        ax.set_facecolor("black")

        col = f"{emotion}_file"
        paths = []
        if col in wide_df.columns:
            for v in wide_df[col].tolist():
                p = _resolve_path(v, image_dir)
                if p is not None:
                    paths.append(p)

        if not paths:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{emotion}\n(n=0)", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=10)
            continue

        density, n_valid = cumulative_density_from_paths(paths, analysis_mask)

        cmap = _emotion_cmap(emotion)
        im = ax.imshow(density, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_axis_off()

        # Emotion name below, in white
        ax.text(
            0.5, -0.06,
            f"{emotion} (n={n_valid})",
            transform=ax.transAxes,
            ha="center", va="top",
            color="white",
            fontsize=11,
        )

        # Optional per-panel colourbar: too busy if you do it for each.
        # We'll add a single shared colourbar for the whole figure instead.

    # Tight spacing: bodies closer together
    plt.subplots_adjust(left=0.01, right=0.92 if add_legend else 0.99, top=0.98, bottom=0.12, wspace=0.01)

    if add_legend:
        # Single shared colourbar at right
        cax = fig.add_axes([0.93, 0.18, 0.015, 0.64])  # [left, bottom, width, height]
        # Use a generic scalar mappable (colourbar just reflects 0..1 proportion)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=_emotion_cmap("Happy"))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("More participants coloured this area", color="white", fontsize=10)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.get_yticklabels(), color="white")
        cb.outline.set_edgecolor("white")

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)


# ------------------------------ NEW: aware↔emotion overlap stats ------------------------------

def compute_aware_emotion_overlap_metrics(
    wide_df: pd.DataFrame,
    analysis_mask: np.ndarray,
    *,
    image_dir: Union[str, os.PathLike, Path],
    out_dir: Union[str, os.PathLike, Path],
) -> pd.DataFrame:
    """
    For each participant and each emotion:
      A = aware mask (binary ink within analysis_mask)
      E = emotion mask (binary ink within analysis_mask)
      Computes overlap metrics and relates to slider later.

    Outputs:
      - out_dir / aware_emotion_overlap_long.csv
    """
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_area = float(analysis_mask.sum())
    rows: List[dict] = []

    for _, row in wide_df.iterrows():
        pid = str(row.get(ID_PRIVATE_COL))

        aware = load_binary_ink_mask_for_participant(row, "Aware", image_dir=image_dir, analysis_mask=analysis_mask)
        if aware is None:
            continue

        aware_sum = float(aware.sum())
        for emotion in EMOTIONS_NO_AWARE:
            em = load_binary_ink_mask_for_participant(row, emotion, image_dir=image_dir, analysis_mask=analysis_mask)
            if em is None:
                continue

            em_sum = float(em.sum())
            inter = float(np.logical_and(aware, em).sum())
            union = float(np.logical_or(aware, em).sum())

            # Slider value for this emotion, if present
            slider_col = SLIDER_EMOTIONS.get(emotion)
            slider_val = pd.to_numeric(row.get(slider_col), errors="coerce") if slider_col else np.nan

            out = {
                ID_PRIVATE_COL: pid,
                "Emotion": emotion,
                "Slider": slider_val,
                "OverlapCount": inter,
                "OverlapFracMask": (inter / mask_area) if mask_area > 0 else np.nan,
                "OverlapFracAware": (inter / aware_sum) if aware_sum > 0 else np.nan,
                "OverlapFracEmotion": (inter / em_sum) if em_sum > 0 else np.nan,
                "Jaccard": (inter / union) if union > 0 else np.nan,
                "AwareInkCount": aware_sum,
                "EmotionInkCount": em_sum,
            }
            rows.append(out)

    long_df = pd.DataFrame(rows)
    long_df.to_csv(out_dir / "aware_emotion_overlap_long.csv", index=False)
    return long_df


def relate_slider_to_overlap(
    overlap_long_df: pd.DataFrame,
    *,
    out_dir: Union[str, os.PathLike, Path],
    metric: str = "Jaccard",
) -> pd.DataFrame:
    """
    Per emotion, relate slider to overlap metric (default: Jaccard).

    Outputs:
      - out_dir / slider_overlap_summary.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if overlap_long_df.empty:
        warnings.warn("Overlap long dataframe is empty; no overlap stats computed.")
        return pd.DataFrame()

    summaries: List[dict] = []
    for emotion, sdf in overlap_long_df.groupby("Emotion", sort=False):
        sdf = sdf.dropna(subset=["Slider", metric])
        n = int(len(sdf))
        row = {"Emotion": emotion, "Metric": metric, "N": n}

        if n < 3:
            row.update({
                "Pearson_r": np.nan, "Pearson_p": np.nan,
                "Spearman_rho": np.nan, "Spearman_p": np.nan,
                "OLS_slope": np.nan, "OLS_intercept": np.nan, "OLS_R2": np.nan, "OLS_p_slope": np.nan,
            })
            summaries.append(row)
            continue

        x = sdf["Slider"].to_numpy(dtype=float)
        y = sdf[metric].to_numpy(dtype=float)

        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y, nan_policy="omit")
        lr = stats.linregress(x, y)

        row.update({
            "Pearson_r": float(pr), "Pearson_p": float(pp),
            "Spearman_rho": float(sr), "Spearman_p": float(sp),
            "OLS_slope": float(lr.slope),
            "OLS_intercept": float(lr.intercept),
            "OLS_R2": float(lr.rvalue ** 2),
            "OLS_p_slope": float(lr.pvalue),
        })
        summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "slider_overlap_summary.csv", index=False)
    return summary_df


# ------------------------------ NEW: slider-weighted aware summary images ------------------------------

def _normalise_weights(values: np.ndarray, mode: str = "minmax", fixed_range: Tuple[float, float] = (0.0, 100.0)) -> np.ndarray:
    """
    mode:
      - "minmax": within-sample min-max -> [0,1]
      - "fixed": clamp to fixed_range then scale to [0,1]
    """
    v = values.astype(float)
    if mode == "minmax":
        if np.all(~np.isfinite(v)):
            return np.zeros_like(v)
        vmin = np.nanmin(v)
        vmax = np.nanmax(v)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)
    elif mode == "fixed":
        lo, hi = fixed_range
        vv = np.clip(v, lo, hi)
        denom = (hi - lo) if hi > lo else 1.0
        return (vv - lo) / denom
    else:
        raise ValueError("weight normalisation mode must be 'minmax' or 'fixed'.")


def compute_slider_weighted_aware_density(
    wide_df: pd.DataFrame,
    analysis_mask: np.ndarray,
    *,
    image_dir: Union[str, os.PathLike, Path],
    emotion: str,
    weight_norm: str = "minmax",
    fixed_range: Tuple[float, float] = (0.0, 100.0),
) -> Tuple[np.ndarray, int]:
    """
    For a given emotion:
      density(x) = sum_i w_i * Aware_i(x) / sum_i w_i
    where w_i is the normalised slider for that emotion and Aware_i is binary aware ink mask.

    Returns:
      density in [0,1] with NaN outside analysis mask, and n_valid participants contributing (w_i>0 and aware exists).
    """
    if emotion not in SLIDER_EMOTIONS:
        raise ValueError(f"Emotion {emotion} has no slider mapping.")

    image_dir = Path(image_dir)
    slider_col = SLIDER_EMOTIONS[emotion]
    sliders = pd.to_numeric(wide_df.get(slider_col), errors="coerce").to_numpy(dtype=float)
    weights = _normalise_weights(sliders, mode=weight_norm, fixed_range=fixed_range)

    H, W = analysis_mask.shape
    num = np.zeros((H, W), dtype=np.float64)
    den = 0.0
    n_valid = 0

    for idx, row in wide_df.iterrows():
        w = float(weights[idx]) if idx < len(weights) else 0.0
        if not np.isfinite(w) or w <= 0:
            continue

        aware = load_binary_ink_mask_for_participant(row, "Aware", image_dir=image_dir, analysis_mask=analysis_mask)
        if aware is None:
            continue

        num += w * aware.astype(np.float64)
        den += w
        n_valid += 1

    if den <= 0:
        density = np.full((H, W), np.nan, dtype=np.float32)
        density[~analysis_mask] = np.nan
        return density, 0

    density = (num / den).astype(np.float32)  # [0,1] weighted probability of being inked
    density[~analysis_mask] = np.nan
    return density, n_valid


def plot_weighted_aware_grid(
    wide_df: pd.DataFrame,
    analysis_mask: np.ndarray,
    *,
    image_dir: Union[str, os.PathLike, Path],
    out_path: Union[str, os.PathLike, Path],
    emotions: Optional[List[str]] = None,
    weight_norm: str = "minmax",
    fixed_range: Tuple[float, float] = (0.0, 100.0),
    dpi: int = 400,
    panel_height_in: float = 3.0,
    add_legend: bool = True,
) -> None:
    """
    Combined grid like the cumulative maps, but each panel is:
      slider-weighted aware density for that emotion.
    """
    if emotions is None:
        emotions = EMOTIONS_NO_AWARE

    image_dir = Path(image_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    K = len(emotions)
    fig_width = panel_height_in * K * 0.75
    fig, axes = plt.subplots(1, K, figsize=(fig_width, panel_height_in), dpi=dpi)
    if K == 1:
        axes = [axes]

    fig.patch.set_facecolor("black")
    norm = Normalize(vmin=0.0, vmax=1.0)

    for ax, emotion in zip(axes, emotions):
        ax.set_facecolor("black")
        try:
            density, n_valid = compute_slider_weighted_aware_density(
                wide_df,
                analysis_mask,
                image_dir=image_dir,
                emotion=emotion,
                weight_norm=weight_norm,
                fixed_range=fixed_range,
            )
        except Exception as e:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{emotion}\n(error)", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=10)
            warnings.warn(str(e))
            continue

        cmap = _emotion_cmap(emotion)
        ax.imshow(density, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_axis_off()
        ax.text(
            0.5, -0.06,
            f"{emotion} (aware weighted; n={n_valid})",
            transform=ax.transAxes,
            ha="center", va="top",
            color="white",
            fontsize=10,
        )

    plt.subplots_adjust(left=0.01, right=0.92 if add_legend else 0.99, top=0.98, bottom=0.12, wspace=0.01)

    if add_legend:
        cax = fig.add_axes([0.93, 0.18, 0.015, 0.64])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=_emotion_cmap("Happy"))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("Higher = more aware shading (weighted by slider)", color="white", fontsize=10)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.get_yticklabels(), color="white")
        cb.outline.set_edgecolor("white")

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)


# ------------------------------ Existing slider↔image linkage ------------------------------

def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def relate_sliders_to_images(
    wide_df: pd.DataFrame,
    *,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: List[pd.DataFrame] = []

    for emotion, slider_col in SLIDER_EMOTIONS.items():
        if slider_col not in wide_df.columns:
            warnings.warn(f"Slider column missing: {slider_col} (emotion={emotion})")
            continue

        slider_vals = _coerce_numeric(wide_df[slider_col])
        ink_frac = _coerce_numeric(wide_df.get(f"{emotion}_InkFrac"))
        ink_count = _coerce_numeric(wide_df.get(f"{emotion}_InkCount"))

        tmp = pd.DataFrame({
            ID_PRIVATE_COL: wide_df[ID_PRIVATE_COL],
            "Emotion": emotion,
            "Slider": slider_vals,
            "InkFrac": ink_frac,
            "InkCount": ink_count,
        })

        tmp = tmp.dropna(subset=["Slider", "InkFrac"])
        long_rows.append(tmp)

    if not long_rows:
        warnings.warn("No slider-image pairs available for linkage.")
        return pd.DataFrame()

    long_df = pd.concat(long_rows, axis=0, ignore_index=True)
    long_df.to_csv(out_dir / "slider_image_long.csv", index=False)

    summaries: List[dict] = []
    for emotion, sdf in long_df.groupby("Emotion", sort=False):
        n = int(len(sdf))
        row = {"Emotion": emotion, "N": n}

        if n < 3:
            row.update({
                "Pearson_r": np.nan, "Pearson_p": np.nan,
                "Spearman_rho": np.nan, "Spearman_p": np.nan,
                "OLS_slope": np.nan, "OLS_intercept": np.nan, "OLS_R2": np.nan, "OLS_p_slope": np.nan,
            })
            summaries.append(row)
            continue

        pr, pp = stats.pearsonr(sdf["Slider"].to_numpy(), sdf["InkFrac"].to_numpy())
        sr, sp = stats.spearmanr(sdf["Slider"].to_numpy(), sdf["InkFrac"].to_numpy(), nan_policy="omit")
        lr = stats.linregress(sdf["Slider"].to_numpy(), sdf["InkFrac"].to_numpy())

        row.update({
            "Pearson_r": float(pr), "Pearson_p": float(pp),
            "Spearman_rho": float(sr), "Spearman_p": float(sp),
            "OLS_slope": float(lr.slope),
            "OLS_intercept": float(lr.intercept),
            "OLS_R2": float(lr.rvalue ** 2),
            "OLS_p_slope": float(lr.pvalue),
        })
        summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "slider_image_summary.csv", index=False)
    return summary_df


# ------------------------------ Main pipeline ------------------------------

def run_pipeline(root_dir: Union[str, os.PathLike]) -> pd.DataFrame:
    root = Path(root_dir)
    image_dir = root / "uploads"
    out_dir = root / "outputs"
    avg_dir = root / "averages"

    if not image_dir.exists():
        raise FileNotFoundError(f"uploads directory not found: {image_dir}")

    analysis_mask = load_analysis_mask("interior_mask.png")
    questionnaire_path = _find_questionnaire(root)

    img_paths = _list_images(image_dir)
    wide = _build_image_table(img_paths, questionnaire_path)
    wide[ID_PRIVATE_COL] = wide[ID_PRIVATE_COL].astype(str)

    # Per-row ink metrics for each emotion image
    for emotion in EMOTIONS:
        col = f"{emotion}_file"
        ink_count_col = f"{emotion}_InkCount"
        ink_frac_col = f"{emotion}_InkFrac"

        ink_counts: List[float] = []
        ink_fracs: List[float] = []

        for v in wide[col].tolist():
            p = _resolve_path(v, image_dir)
            if p is None:
                ink_counts.append(np.nan)
                ink_fracs.append(np.nan)
                continue
            try:
                m = ink_metrics_for_image(p, analysis_mask)
                ink_counts.append(m["ink_count"])
                ink_fracs.append(m["ink_frac"])
            except Exception as e:
                warnings.warn(f"Failed metrics for {p}: {e}")
                ink_counts.append(np.nan)
                ink_fracs.append(np.nan)

        wide[ink_count_col] = ink_counts
        wide[ink_frac_col] = ink_fracs

    out_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_dir / "participants_wide_with_ink_metrics.csv", index=False)

    # Existing slider↔matched emotion ink linkage
    summary_df = relate_sliders_to_images(wide, out_dir=out_dir)

    # NEW: styled combined cumulative plot
    avg_dir.mkdir(parents=True, exist_ok=True)
    plot_emotion_cumulative_grid_styled(
        wide_df=wide,
        analysis_mask=analysis_mask,
        image_dir=image_dir,
        out_path=avg_dir / "all_emotions_compiled_styled.png",
        dpi=450,
        panel_height_in=3.0,
        add_legend=True,
    )

    # NEW: aware↔emotion overlap metrics and slider linkage
    overlap_long = compute_aware_emotion_overlap_metrics(
        wide_df=wide,
        analysis_mask=analysis_mask,
        image_dir=image_dir,
        out_dir=out_dir,
    )
    relate_slider_to_overlap(overlap_long, out_dir=out_dir, metric="OverlapFracMask")

    # NEW: slider-weighted aware summaries
    plot_weighted_aware_grid(
        wide_df=wide,
        analysis_mask=analysis_mask,
        image_dir=image_dir,
        out_path=avg_dir / "aware_weighted_by_slider_compiled.png",
        weight_norm="minmax",  # switch to "fixed" if you want fixed instrument scale
        fixed_range=(0.0, 100.0),
        dpi=450,
        panel_height_in=3.0,
        add_legend=True,
    )

    # Console sanity
    if not summary_df.empty:
        print("\n=== Slider ↔ Emotion InkFrac (matched emotion) ===")
        print(summary_df.to_string(index=False))
    if not overlap_long.empty:
        print("\n=== Aware overlap table written to outputs/aware_emotion_overlap_long.csv ===")

    return wide


if __name__ == "__main__":
    run_pipeline("body_map_data")
