# ISO-Finder by Alexander T. Engelbrecht (xelaeris) C www.hof.technology

  a CLI tool) called "LSST_CATX_ISO_FINDER_v0.2".
Purpose: detect and score possible interstellar-object–like features (jets/tails) in astronomical images, and do a simple RGB-based spectral "seed" classification (dust / gas/ion / mixed / unknown).
Input: R-band image (required) and optional G/B images (or FITS). Parameters: pixel scale (hardcoded fallback 1.38 arcsec/px), geocentric distance in AU, optional sun angle in image degrees.
Output: an ISOMetrics dataclass instance and a brief printed report (length, collimation/JCI, sunward flag, spectral seed, K_ISO score, reason codes).
Main components and how they work

ISOMetrics dataclass: holds output/metrics (file id, pixel scale in km, length/width in km, collimation, linearity, is_sunward flag, jet_collimation_index, seed_type, delta_R/delta_B, k_iso score, reason codes).
Utilities:
robust_normalize(img): scales image to 0..1 using 1st and 99.5th percentiles (guard against flat or NaN images).
load_image(path): reads FITS (using astropy) or other image formats via imageio.v3; returns float32 array.
_robust_mean(x): median/MAD-based clipped mean to reduce outlier influence.
Spectral logic: spectral_gradient_proxy(r,g,b,jet_mask,coma_mask)
Computes robust mean intensities inside jet_mask and coma_mask for each channel.
Normalizes per-patch sums to compute fractions jR/jB and cR/cB.
dR = jR - cR, dB = jB - cB (differences between jet and coma color fractions).
Classification thresholds:
TH = 0.06
If dB > TH and dR < -TH/2 -> GAS_ION
If dR > TH and dB < -TH/2 -> DUST
If |dR| or |dB| > TH/2 -> MIXED
Otherwise -> UNKNOWN (and returns reason codes for failure modes)
Returns status, reason, dR, dB.
Core analysis: analyze_comet(img_r, img_g, img_b, px_scale_arcsec, dist_au, sun_position_deg)
Computes physical pixel scale in km using distance (AU -> km) and small-angle conversion.
Builds segmentation masks from R-band:
Normalizes R image and computes Otsu threshold.
Coma mask: pixels above 1.5 * Otsu threshold; largest connected component taken as core/coma; creates circular-ish mask around its centroid for spectral reference.
Jet mask: (norm > 0.7 * Otsu) and not in coma; removes small objects (min area 20 px); selects largest connected component as the jet.
If no jet found -> returns failure ISOMetrics with reason FAIL_NO_TAIL.
Measures jet_blob properties (major/minor axes, eccentricity, centroid).
length_km, width_km from axes * px_scale_km
collimation uses jet_blob.eccentricity (close to 1 for a line)
Sunward check: if sun_position_deg provided, computes vector from coma center to jet centroid, angle_jet and compares to sun_position_deg:
diff < 30 deg -> is_sunward True (reason PASS_SUNWARD_JET)
diff > 150 deg -> is_sunward False (INFO_NORMAL_TAIL)
otherwise INFO_OFF_AXIS_JET
NOTE: this is a crude image-frame check — not WCS-aware.
Calls spectral_gradient_proxy and appends spectral reason codes.
Computes K_ISO score: base 0.2 +0.3 if collimation>0.9 +0.3 if is_sunward True +0.2 if seed_type != UNKNOWN. If score >= 0.7 adds VERDICT_ISO_ANOMALY.
Returns populated ISOMetrics.
CLI entrypoint / running

The notebook code includes a main() function and if name == "main": main() guard, so it can run as a script.
Usage printed in the header and via argparse:
Example: python3 lsst_catx_iso_finder_v0_2.py --r comet_r.fits --g comet_g.fits --b comet_b.fits --dist-au 1.2 --sun-deg 45
In Colab you can open the notebook with the provided badge link.
Dependencies and environment

numpy, scipy (ndimage), scikit-image (filters, morphology, measure), imageio.v3
astropy optional (for FITS): code checks for astropy and falls back if not installed but then won't read FITS properly.
Tested as a script but currently embedded into a notebook cell.
Limitations, caveats and failure modes

Assumes input channels are registered/aligned and same shape — otherwise returns FAIL_SHAPE_MISMATCH from spectral logic.
Pixel scale is hardcoded fallback (1.38 arcsec/px). Ideally read from FITS header WCS for accurate km scaling.
Sunward check is image-frame only and not WCS-aware — needs calibration to map celestial angles.
Thresholding choices (Otsu * 1.5, *0.7) and spectral thresholds (TH=0.06) are heuristic and might need tuning for different sensors/SNR.
Low SNR or missing color channels -> spectral result UNKNOWN or FAIL_LOW_SNR_SPECTRAL.
Jet detection fails if tail is too faint or fragmented (min area 20 px might need changing).
Using eccentricity as sole collimation metric is simplistic; could use skeletonization + linear fit, or Hough transform for more robust linearity.
Quick suggestions for improvement

Read pixel scale and orientation from FITS/WCS headers when available.
Use background estimation and PSF-aware detection (e.g., aperture photometry, matched filtering) before segmentation.
Use morphological skeletonization and line-fitting to compute a more robust Jet Collimation Index (JCI).
Add SNR thresholds and error propagation for spectral deltas, and bootstrap uncertainties.
Make spectral thresholds configurable via CLI flags.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSST_CATX_ISO_FINDER_v0.2 — Interstellar Object / Comet Audit (Spectral Upgrade)
- Core: Collimation (Anti-Tail), Fragmentation, Non-Gravitational Signs
- Upgrade: Spectral Gradient Proxy (R/G/B) -> Seed Classification (Dust vs Gas/Ion)
- Usage: python3 lsst_catx_iso_finder_v0_2.py --r comet_r.fits --g comet_g.fits --b comet_b.fits --dist-au 1.2

EKE-Changes:
- Robust Main/Argparse
- Integrated spectral_gradient_proxy
- JCI (Jet Collimation Index) added
"""

from __future__ import annotations

import argparse
import sys
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    binary_opening,
    skeletonize,
    disk,
)
from skimage.measure import label, regionprops

try:
    from astropy.io import fits
except ImportError:
    fits = None
import imageio.v3 as iio


@dataclass
class ISOMetrics:
    file_id: str
    px_scale_km: float

    length_km: float
    width_km: float
    collimation_factor: float
    linearity: float
    
    # Jet Logic
    is_sunward: Optional[bool]
    jet_collimation_index: float # JCI (0..1)

    # Spectral
    seed_type: str  # DUST, GAS_ION, MIXED, UNKNOWN
    delta_R: float
    delta_B: float

    k_iso: float
    reason_codes: List[str]


# -----------------------------
# Utils
# -----------------------------
def robust_normalize(img: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(img, [1.0, 99.5])
    if hi <= lo:
        return np.nan_to_num(img).astype(np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

def load_image(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    try:
        if path.lower().endswith((".fits", ".fit")):
            if fits:
                with fits.open(path) as hdul:
                    # heuristic: find first 2D
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim >= 2:
                            d = hdu.data
                            if d.ndim > 2: d = d[0]
                            return d.astype(np.float32)
        return iio.imread(path).astype(np.float32)
    except Exception as e:
        print(f"WARN: Could not load {path}: {e}")
        return None

def _robust_mean(x: np.ndarray, clip_sigma: float = 3.0) -> float:
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    sigma = 1.4826 * mad
    lo, hi = med - clip_sigma * sigma, med + clip_sigma * sigma
    x2 = x[(x >= lo) & (x <= hi)]
    return float(np.mean(x2)) if x2.size else float("nan")

# -----------------------------
# Spectral Logic
# -----------------------------
def spectral_gradient_proxy(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    jet_mask: np.ndarray,
    coma_mask: np.ndarray,
    eps: float = 1e-9
) -> Dict[str, Any]:
    """
    Compares Jet vs Coma in R/G/B to classify seed type.
    """
    if r is None or g is None or b is None:
        return {"status": "UNKNOWN", "reason": "INFO_NO_COLOR_DATA", "dR": 0.0, "dB": 0.0}

    # Ensure shapes match (simple crop/resize check omitted for EKE brevity, assuming registered)
    if r.shape != g.shape or r.shape != b.shape:
        return {"status": "UNKNOWN", "reason": "FAIL_SHAPE_MISMATCH", "dR": 0.0, "dB": 0.0}

    jr, jg, jb = (_robust_mean(r[jet_mask]), _robust_mean(g[jet_mask]), _robust_mean(b[jet_mask]))
    cr, cg, cb = (_robust_mean(r[coma_mask]), _robust_mean(g[coma_mask]), _robust_mean(b[coma_mask]))

    if not np.isfinite([jr, jg, jb, cr, cg, cb]).all():
        return {"status": "UNKNOWN", "reason": "FAIL_LOW_SNR_SPECTRAL", "dR": 0.0, "dB": 0.0}

    jsum = jr + jg + jb + eps
    csum = cr + cg + cb + eps

    jR, jB = jr / jsum, jb / jsum
    cR, cB = cr / csum, cb / csum

    dR, dB = jR - cR, jB - cB
    
    # Classification Logic
    TH = 0.06
    seed = "UNKNOWN"
    reason = "FAIL_UNCERTAIN_SPECTRAL"
    
    if dB > TH and dR < -TH/2:
        seed = "GAS_ION"
        reason = "PASS_SPECTRAL_GAS_ION"
    elif dR > TH and dB < -TH/2:
        seed = "DUST"
        reason = "PASS_SPECTRAL_DUST"
    elif (abs(dR) > TH/2) or (abs(dB) > TH/2):
        seed = "MIXED"
        reason = "PASS_SPECTRAL_MIXED"

    return {"status": seed, "reason": reason, "dR": dR, "dB": dB}

# -----------------------------
# Core Analysis
# -----------------------------
def analyze_comet(
    img_r: np.ndarray,
    img_g: Optional[np.ndarray],
    img_b: Optional[np.ndarray],
    px_scale_arcsec: float,
    dist_au: float,
    sun_position_deg: Optional[float]
) -> ISOMetrics:
    
    reason = []
    
    # Physics Scale
    dist_km = dist_au * 1.496e8
    # tan(1 arcsec) ~ 4.848e-6
    px_scale_km = (dist_km * 4.848e-6) * px_scale_arcsec

    # 1. Segment Coma vs Jet
    norm = robust_normalize(img_r)
    thresh = threshold_otsu(norm)
    
    # Core mask (Coma approx)
    coma_thresh = thresh * 1.5
    mask_coma_raw = norm > coma_thresh
    # Keep largest blob as nucleus/coma
    lbl_c = label(mask_coma_raw)
    if lbl_c.max() > 0:
        props_c = regionprops(lbl_c)
        props_c.sort(key=lambda x: x.area, reverse=True)
        coma_blob = props_c[0]
        # Create circular-ish mask around centroid for spectral check
        cy, cx = coma_blob.centroid
        y, x = np.ogrid[:norm.shape[0], :norm.shape[1]]
        # Radius ~ 1.5 * major axis of core blob
        r_coma = 1.5 * (coma_blob.major_axis_length / 2.0)
        mask_coma = (x - cx)**2 + (y - cy)**2 <= r_coma**2
    else:
        # Fallback
        mask_coma = mask_coma_raw

    # Jet mask (Tail) - Faint stuff
    mask_jet_raw = (norm > (thresh * 0.7)) & (~mask_coma)
    mask_jet_raw = remove_small_objects(mask_jet_raw, 20)
    
    lbl_j = label(mask_jet_raw)
    if lbl_j.max() == 0:
        return ISOMetrics("err", 0,0,0,0,0,None,0,"UNKNOWN",0,0,0,["FAIL_NO_TAIL"])
    
    props_j = regionprops(lbl_j)
    props_j.sort(key=lambda x: x.area, reverse=True)
    jet_blob = props_j[0]
    mask_jet = lbl_j == jet_blob.label

    # Metrics
    major = jet_blob.major_axis_length
    minor = jet_blob.minor_axis_length
    length_km = major * px_scale_km
    width_km = minor * px_scale_km
    
    aspect = major / (minor + 1e-6)
    collimation = jet_blob.eccentricity  # 1.0 = line
    
    if collimation > 0.9:
        reason.append("PASS_HIGH_COLLIMATION")
    
    # Sunward Check
    is_sunward = None
    if sun_position_deg is not None:
        orientation = np.degrees(jet_blob.orientation)
        # Orientation is usually -90..90 or 0..180 depending on impl.
        # skimage: -pi/2 to pi/2. Convert to 0..360 frame roughly?
        # Robust check: Vector from Coma Center to Jet Centroid
        if 'cy' in locals():
            jy, jx = jet_blob.centroid
            dy, dx = jy - cy, jx - cx
            angle_jet = math.degrees(math.atan2(-dy, dx)) # Image coords y down? assume standard math
            # This needs calibration with WCS ideally. For now: raw check.
            # Assuming sun_position_deg is in same frame.
            diff = abs(angle_jet - sun_position_deg)
            while diff > 180: diff = abs(diff - 360)
            
            if diff < 30:
                is_sunward = True
                reason.append("PASS_SUNWARD_JET")
            elif diff > 150:
                is_sunward = False
                reason.append("INFO_NORMAL_TAIL")
            else:
                reason.append("INFO_OFF_AXIS_JET")

    # Spectral Audit
    spec_res = spectral_gradient_proxy(img_r, img_g, img_b, mask_jet, mask_coma)
    seed_type = spec_res["status"]
    reason.append(spec_res["reason"])

    # Score K_ISO
    # Base 0.2
    # Collimation > 0.9 -> +0.3
    # Sunward -> +0.3
    # Seed Type Known -> +0.2
    score = 0.2
    if collimation > 0.9: score += 0.3
    if is_sunward: score += 0.3
    if seed_type != "UNKNOWN": score += 0.2
    
    # JCI
    jci = collimation

    if score >= 0.7:
        reason.append("VERDICT_ISO_ANOMALY")
    
    return ISOMetrics(
        file_id="current_img",
        px_scale_km=px_scale_km,
        length_km=length_km,
        width_km=width_km,
        collimation_factor=collimation,
        linearity=collimation,
        is_sunward=is_sunward,
        jet_collimation_index=jci,
        seed_type=seed_type,
        delta_R=spec_res["dR"],
        delta_B=spec_res["dB"],
        k_iso=score,
        reason_codes=sorted(list(set(reason)))
    )


def main():
    ap = argparse.ArgumentParser(description="LSST_CATX_ISO_FINDER v0.2")
    ap.add_argument("--r", required=True, help="Path to R-band image")
    ap.add_argument("--g", help="Path to G-band image")
    ap.add_argument("--b", help="Path to B-band image")
    ap.add_argument("--dist-au", type=float, default=1.0, help="Geocentric distance in AU")
    ap.add_argument("--sun-deg", type=float, help="Sun angle in image deg (0=East, 90=North)")
    args = ap.parse_args()
    
    img_r = load_image(args.r)
    if img_r is None:
        print(f"ERROR: Could not load R-band image: {args.r}")
        sys.exit(1)
        
    img_g = load_image(args.g)
    img_b = load_image(args.b)
    
    # EKE: Hardcoded pixel scale from prompt if not in header (1.38 arcsec/px)
    # Ideally read from header
    px_scale = 1.38 
    
    m = analyze_comet(img_r, img_g, img_b, px_scale, args.dist_au, args.sun_deg)
    
    print("-" * 40)
    print(f"ISO-FINDER v0.2 REPORT")
    print(f"Object Scale: {m.length_km:.1e} km")
    print(f"Collimation (JCI): {m.jet_collimation_index:.3f}")
    print(f"Sunward Jet: {m.is_sunward}")
    print(f"Spectral Seed: {m.seed_type} (dR={m.delta_R:.2f}, dB={m.delta_B:.2f})")
    print("-" * 40)
    print(f"K_ISO Score: {m.k_iso:.2f}")
    print(f"Reasons: {', '.join(m.reason_codes)}")
    print("-" * 40)

if __name__ == "__main__":
    main()

