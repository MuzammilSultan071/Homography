#!/usr/bin/env python3
"""
document_scanner.py
===================

Classical OpenCV document scanner with robust automatic corner detection
and automatic orientation correction.

Detection strategy
------------------
Binarization masks tried:
    * Canny edges
    * Adaptive Gaussian threshold (inverted)
    * Otsu (both polarities)
    * HSV "paper" mask         (bright + low saturation)
    * HSV "paper" strict mask

Each contour -> quad via approxPolyDP sweep, convex hull, or minAreaRect.
Each quad scored by:
    * area fraction
    * parallelogram-ness
    * right-angle regularity
    * paper-ness of the interior   (bright + desaturated)
    * border-touch penalty         (rejects the image frame)

Post-warp orientation
---------------------
After warping, the dominant direction of long straight lines is measured
via Hough transform. If lines are mostly vertical, the page is rotated
90 degrees and is auto-rotated CCW. Use --rotate {auto,0,90,180,270}
to override.

Usage
-----
    python document_scanner.py input.jpg -o scan.png --a4 --dpi 300
    python document_scanner.py input.jpg -o scan.png --debug
    python document_scanner.py input.jpg -o scan.png --rotate 90
    python document_scanner.py input.jpg -o scan.png --rotate 0     # disable auto
    python document_scanner.py input.jpg -o scan.png --corners x1,y1,...,x4,y4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# ===========================================================================
# 1. I/O
# ===========================================================================

def load_image(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input image not found: {path}")
    if _HAS_PIL:
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not decode image: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(path)[1].lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cv2.imencode failed for extension {ext!r}")
    buf.tofile(path)


def _dump(debug_dir: str, name: str, img: np.ndarray) -> None:
    try:
        imwrite_unicode(os.path.join(debug_dir, name), img)
    except Exception as exc:
        print(f"[warn] could not write debug image {name}: {exc}")


# ===========================================================================
# 2. Utilities
# ===========================================================================

def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def auto_canny_thresholds(gray: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    v = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    if hi <= lo:
        lo, hi = 50, 150
    return lo, hi


def _signed_area(pts: np.ndarray) -> float:
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    pts = pts[np.argsort(angles)]
    sums = pts.sum(axis=1)
    start = int(np.argmin(sums))
    pts = np.roll(pts, -start, axis=0)
    if _signed_area(pts) < 0:
        pts = pts[[0, 3, 2, 1]]
    return pts.astype(np.float32)


# ===========================================================================
# 3. HSV paper mask
# ===========================================================================

def _paper_mask(image: np.ndarray, v_thresh: int, s_thresh: int) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    return ((v > v_thresh) & (s < s_thresh)).astype(np.uint8) * 255


# ===========================================================================
# 4. Quad extraction + scoring
# ===========================================================================

def _quad_from_contour(contour: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(contour, True)
    if peri < 80:
        return None
    if len(contour) > 30:
        contour = cv2.approxPolyDP(contour, 0.005 * peri, True)
    for eps in (0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05,
                0.06, 0.07, 0.08, 0.10, 0.12, 0.15):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _paper_ness(quad: np.ndarray, image: np.ndarray) -> float:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    erode = max(5, int(min(h, w) * 0.02)) | 1
    mask = cv2.erode(mask, np.ones((erode, erode), np.uint8))
    if int(mask.sum()) == 0:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    valid = mask > 0
    paper = (v[valid] > 140) & (s[valid] < 70)
    return float(paper.mean())


def _border_touch_count(quad: np.ndarray, shape: Tuple[int, int],
                        tol: int = 6) -> int:
    h, w = shape[:2]
    n = 0
    for x, y in quad:
        if x <= tol or x >= w - 1 - tol or y <= tol or y >= h - 1 - tol:
            n += 1
    return n


def _score_quad(quad: np.ndarray, image: np.ndarray, image_area: float) -> float:
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    area = abs(float(cv2.contourArea(pts)))
    if area < 0.03 * image_area or area > 0.998 * image_area:
        return -1.0
    sides = np.array([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)])
    if sides.min() < 15:
        return -1.0
    sr1 = min(sides[0], sides[2]) / max(sides[0], sides[2])
    sr2 = min(sides[1], sides[3]) / max(sides[1], sides[3])
    par = 0.5 * (sr1 + sr2)
    angles = []
    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            return -1.0
        c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angles.append(np.arccos(c))
    dev = float(np.mean([abs(a - np.pi / 2) for a in angles]))
    ang = max(0.0, 1.0 - dev / (np.pi / 2))
    geometry = (area / image_area) * (0.30 + 0.40 * par + 0.30 * ang)
    paper = _paper_ness(pts, image)
    touches = _border_touch_count(pts, image.shape[:2])
    border_factor = 1.0 if touches <= 1 else 0.45 ** (touches - 1)
    return geometry * (0.10 + 0.90 * paper) * border_factor


# ===========================================================================
# 5. Candidate collection
# ===========================================================================

def _collect_candidates(
    gray: np.ndarray,
    color: np.ndarray,
    image_area: float,
    debug_dir: Optional[str] = None,
) -> List[Tuple[np.ndarray, float, str, np.ndarray]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates: List[Tuple[np.ndarray, float, str, np.ndarray]] = []
    seen = set()

    def _add(quad: np.ndarray, contour: np.ndarray, label: str) -> None:
        s = _score_quad(quad, color, image_area)
        if s <= 0:
            return
        key = (label.split("+")[0], round(abs(cv2.contourArea(quad)) / 100))
        if key in seen:
            return
        seen.add(key)
        candidates.append((quad, s, label, contour))

    def _process_mask(mask: np.ndarray, label: str) -> None:
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.dilate(m, kernel, iterations=1)
        all_contours: List[np.ndarray] = []
        for mode in (cv2.RETR_EXTERNAL, cv2.RETR_LIST):
            cs, _ = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(cs)
        all_contours.sort(key=cv2.contourArea, reverse=True)
        for c in all_contours[:30]:
            a = cv2.contourArea(c)
            if a < 0.03 * image_area:
                continue
            q = _quad_from_contour(c)
            if q is not None:
                _add(q, c, label)
                continue
            hull = cv2.convexHull(c)
            q = _quad_from_contour(hull)
            if q is not None:
                _add(q, c, label + "+hull")
                continue
            rect = cv2.minAreaRect(c)
            if rect[1][0] > 5 and rect[1][1] > 5:
                box = cv2.boxPoints(rect).astype(np.float32)
                ra = rect[1][0] * rect[1][1]
                if ra > 0 and (a / ra) > 0.75:
                    _add(box, c, label + "+minrect")

    # Canny
    lo, hi = auto_canny_thresholds(blur)
    edges = cv2.Canny(blur, lo, hi)
    _process_mask(edges, "canny")
    if debug_dir:
        _dump(debug_dir, "10_canny.png", edges)

    # Adaptive
    ad = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    _process_mask(ad, "adaptive")
    if debug_dir:
        _dump(debug_dir, "11_adaptive.png", ad)

    # Otsu both polarities
    _, otsu = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _process_mask(otsu, "otsu")
    _process_mask(cv2.bitwise_not(otsu), "otsu_inv")
    if debug_dir:
        _dump(debug_dir, "12_otsu.png", otsu)

    # HSV paper
    paper = _paper_mask(color, 130, 90)
    _process_mask(paper, "paper")
    if debug_dir:
        _dump(debug_dir, "13_paper.png", paper)

    paper_strict = _paper_mask(color, 170, 55)
    _process_mask(paper_strict, "paper_strict")
    if debug_dir:
        _dump(debug_dir, "14_paper_strict.png", paper_strict)

    return candidates


def detect_document_quad(
    image: np.ndarray,
    *,
    min_area_ratio: float = 0.05,
    canny_sigma: float = 0.33,
    debug_dir: Optional[str] = None,
    verbose: bool = True,
    downscale_max: int = 1000,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    gray = to_gray(image)
    h, w = gray.shape[:2]

    scale = 1.0
    if max(h, w) > downscale_max:
        scale = downscale_max / float(max(h, w))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        small_gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        small_color = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray
        small_color = image

    image_area = float(small_gray.shape[0] * small_gray.shape[1])
    candidates = _collect_candidates(small_gray, small_color,
                                     image_area, debug_dir)

    if not candidates:
        if verbose:
            print("[warn] No quad candidates found.")
        return None, None

    candidates.sort(key=lambda t: -t[1])
    if verbose:
        print(f"[info] Detection candidates ({len(candidates)}):")
        for q, s, label, _ in candidates[:6]:
            paper = _paper_ness(q, small_color)
            touches = _border_touch_count(q, small_gray.shape[:2])
            print(f"       {label:<20} score={s:.3f}  "
                  f"area={cv2.contourArea(q):.0f}  "
                  f"paper={paper:.2f}  border_touch={touches}")

    best_quad_small, best_score, best_label, best_contour = candidates[0]
    if verbose:
        print(f"[info] Best: {best_label} (score={best_score:.3f})")

    if cv2.contourArea(best_quad_small) < min_area_ratio * image_area:
        if verbose:
            print("[warn] Best candidate below --min-area-ratio; rejecting.")
        return None, None

    quad = best_quad_small / scale
    quad = order_points(quad)
    contour = (best_contour.astype(np.float32) / scale).astype(np.int32)
    return quad, contour


# ===========================================================================
# 6. Sub-pixel refinement
# ===========================================================================

def _intersect_lines(l1, l2) -> Optional[np.ndarray]:
    x1, y1, vx1, vy1 = l1
    x2, y2, vx2, vy2 = l2
    den = vx1 * vy2 - vy1 * vx2
    if abs(den) < 1e-9:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / den
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)


def refine_quad_with_lines(
    quad: np.ndarray,
    contour: np.ndarray,
    image_shape: Tuple[int, ...],
    max_shift_ratio: float = 0.05,
) -> Optional[np.ndarray]:
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 12:
        return None
    diag = float(np.hypot(image_shape[0], image_shape[1]))
    max_shift = max_shift_ratio * diag
    lines = []
    for i in range(4):
        p = quad[i]
        q = quad[(i + 1) % 4]
        seg = q - p
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            return None
        dist = np.abs(seg[0] * (pts[:, 1] - p[1])
                      - seg[1] * (pts[:, 0] - p[0])) / length
        tol = max(3.0, 0.03 * length)
        sel = pts[dist < tol]
        if len(sel) < 10:
            return None
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((float(x0), float(y0), float(vx), float(vy)))
    new_pts = []
    for i in range(4):
        pt = _intersect_lines(lines[(i - 1) % 4], lines[i])
        if pt is None:
            return None
        new_pts.append(pt)
    new_quad = np.asarray(new_pts, dtype=np.float32)
    if not cv2.isContourConvex(new_quad.reshape(-1, 1, 2).astype(np.int32)):
        return None
    if float(np.max(np.linalg.norm(new_quad - quad, axis=1))) > max_shift:
        return None
    a_old = abs(float(cv2.contourArea(quad.astype(np.float32))))
    a_new = abs(float(cv2.contourArea(new_quad)))
    if a_old > 0 and a_new < 0.5 * a_old:
        return None
    return order_points(new_quad)


# ===========================================================================
# 7. Homography + warp
# ===========================================================================

A4_W_MM, A4_H_MM = 210.0, 297.0


def compute_output_size(
    quad: Optional[np.ndarray] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    a4: bool = False,
    dpi: int = 300,
) -> Tuple[int, int]:
    if a4:
        if width and height:
            return int(width), int(height)
        if width:
            return int(width), int(round(width * A4_H_MM / A4_W_MM))
        if height:
            return int(round(height * A4_W_MM / A4_H_MM)), int(height)
        return (int(round(A4_W_MM / 25.4 * dpi)),
                int(round(A4_H_MM / 25.4 * dpi)))
    W = H = None
    if quad is not None:
        tl, tr, br, bl = order_points(quad)
        W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width:
        W = int(width)
    if height:
        H = int(height)
    if W is None or H is None:
        raise ValueError("Cannot determine output size; pass --width/--height.")
    return max(W, 1), max(H, 1)


def four_point_transform(
    image: np.ndarray,
    quad: np.ndarray,
    out_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> Tuple[np.ndarray, np.ndarray]:
    quad = order_points(quad)
    W, H = int(out_size[0]), int(out_size[1])
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                   dtype=np.float32)
    H_mat = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, H_mat, (W, H),
                                 flags=interpolation, borderMode=border_mode)
    return warped, H_mat


# ===========================================================================
# 8. Auto-orientation of the warped page
# ===========================================================================

def detect_line_orientation(gray: np.ndarray) -> int:
    """
    Detect whether the dominant document lines are horizontal or vertical.
    Returns 90 when the page should be rotated 90 degrees CCW,
    otherwise returns 0.
    """
    h, w = gray.shape[:2]

    if max(h, w) > 900:
        s = 900.0 / max(h, w)
        gray = cv2.resize(
            gray, None, fx=s, fy=s,
            interpolation=cv2.INTER_AREA
        )
        h, w = gray.shape[:2]

    _, otsu = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    adapt = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        15
    )

    binary = cv2.bitwise_or(otsu, adapt)

    min_len = max(40, min(h, w) // 4)

    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180.0,
        threshold=max(40, min(h, w) // 12),
        minLineLength=min_len,
        maxLineGap=12
    )

    if lines is None:
        return 0

    angle_hist = np.zeros(180, dtype=np.float64)

    for line in lines:
        values = np.asarray(line).reshape(-1)

        if len(values) < 4:
            continue

        x1, y1, x2, y2 = map(float, values[:4])

        dx = x2 - x1
        dy = y2 - y1

        length = float(np.hypot(dx, dy))

        if length < 1:
            continue

        angle = int(
            round(np.degrees(np.arctan2(dy, dx)))
        ) % 180

        angle_hist[angle] += length

    horiz = (
        angle_hist[0:25].sum()
        + angle_hist[155:180].sum()
    )

    vert = angle_hist[65:115].sum()

    if vert > horiz * 1.2 and vert > 0:
        return 90

    return 0

def auto_orient_warped(warped: np.ndarray, verbose: bool = True
                       ) -> Tuple[np.ndarray, int]:
    """Rotate the warped page 90 deg CCW if its long lines are vertical."""
    angle = detect_line_orientation(to_gray(warped))
    if angle == 90:
        if verbose:
            print("[info] Auto-rotating page 90 deg CCW "
                  "(detected vertical document lines).")
        return cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE), 90
    if verbose:
        print("[info] Orientation OK (document lines are horizontal).")
    return warped, 0


# ===========================================================================
# 9. Post-processing
# ===========================================================================

def _background_estimate(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = min(1.0, 500.0 / float(max(h, w)))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    k = max(3, min(101, int(min(sw, sh) * 0.08) | 1))
    bg_small = cv2.medianBlur(small, k)
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize_illumination(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        bg = np.maximum(_background_estimate(img), 1)
        return cv2.divide(img, bg, scale=255)
    out = []
    for ch in cv2.split(img):
        bg = np.maximum(_background_estimate(ch), 1)
        out.append(cv2.divide(ch, bg, scale=255))
    return cv2.merge(out)


# ===========================================================================
# 10. CLI
# ===========================================================================

def parse_corners(text: str) -> np.ndarray:
    vals = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if len(vals) != 8:
        raise ValueError("--corners needs 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4")
    return np.array(vals, dtype=np.float32).reshape(4, 2)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classical OpenCV document scanner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="input photo")
    p.add_argument("-o", "--output", default="scan.png", help="output path")

    g = p.add_argument_group("output size")
    g.add_argument("--width", type=int, default=None)
    g.add_argument("--height", type=int, default=None)
    g.add_argument("--a4", action="store_true")
    g.add_argument("--dpi", type=int, default=300)

    g = p.add_argument_group("detection tuning")
    g.add_argument("--min-area-ratio", type=float, default=0.05)
    g.add_argument("--canny-sigma", type=float, default=0.33)
    g.add_argument("--no-refine", action="store_true")
    g.add_argument("--corners", type=str, default=None)

    g = p.add_argument_group("orientation")
    g.add_argument("--rotate", default="auto",
                   choices=["auto", "0", "90", "180", "270"],
                   help="rotate the warped page. 'auto' detects line "
                        "orientation; numeric values force a rotation "
                        "in degrees clockwise.")

    g = p.add_argument_group("appearance")
    g.add_argument("--mode", choices=["color", "gray", "bw"], default="color")
    g.add_argument("--enhance", action="store_true")

    g = p.add_argument_group("debug")
    g.add_argument("--debug", action="store_true")
    g.add_argument("--debug-dir", default=None)
    g.add_argument("--show", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    image = load_image(args.input)
    print(f"[info] Loaded {args.input}  ({image.shape[1]}x{image.shape[0]})")

    debug_dir = None
    if args.debug:
        debug_dir = args.debug_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "debug"
        )
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[info] Debug images -> {debug_dir}")

    contour = None
    if args.corners:
        quad = parse_corners(args.corners)
        print("[info] Using user-supplied corners (detection skipped).")
    else:
        quad, contour = detect_document_quad(
            image,
            min_area_ratio=args.min_area_ratio,
            canny_sigma=args.canny_sigma,
            debug_dir=debug_dir,
        )
        if quad is None:
            h, w = image.shape[:2]
            quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                            dtype=np.float32)
            print("[warn] Falling back to the whole image as the document quad.")
        if not args.no_refine and contour is not None:
            refined = refine_quad_with_lines(quad, contour, image.shape)
            if refined is not None:
                quad = refined
                print("[info] Corners refined with line fitting.")

    quad = order_points(quad)
    print("[info] Final quad (TL, TR, BR, BL):")
    for name, (x, y) in zip(("TL", "TR", "BR", "BL"), quad):
        print(f"       {name}: ({x:.1f}, {y:.1f})")

    if debug_dir:
        vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 0, 255), 3)
        for i, (x, y) in enumerate(quad):
            cv2.circle(vis, (int(x), int(y)), 9, (0, 255, 255), -1)
            cv2.putText(vis, ("TL", "TR", "BR", "BL")[i],
                        (int(x) + 12, int(y) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3,
                        cv2.LINE_AA)
        _dump(debug_dir, "07_detected_corners.png", vis)

    W, H = compute_output_size(quad, args.width, args.height, args.a4, args.dpi)
    print(f"[info] Output size: {W} x {H}")

    warped, H_mat = four_point_transform(image, quad, (W, H))
    print("[info] Homography H =")
    print(np.array2string(H_mat, precision=6, suppress_small=True))

    if debug_dir:
        _dump(debug_dir, "08_warped_raw.png", warped)

    # ------------------------------------------------------ orientation fix
    if args.rotate == "auto":
        warped, _ = auto_orient_warped(warped)
    else:
        deg = int(args.rotate)
        if deg == 90:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            print("[info] Forced rotation: 90 deg CW")
        elif deg == 180:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
            print("[info] Forced rotation: 180 deg")
        elif deg == 270:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            print("[info] Forced rotation: 90 deg CCW")

    if debug_dir:
        _dump(debug_dir, "08b_warped_oriented.png", warped)

    # ------------------------------------------------------------ appearance
    out = warped
    if args.enhance or args.mode == "bw":
        out = normalize_illumination(out)
    if args.mode == "gray":
        out = to_gray(out)
    elif args.mode == "bw":
        g = to_gray(out)
        block = max(15, (min(g.shape[:2]) // 20) | 1)
        out = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block, 15)

    if debug_dir:
        _dump(debug_dir, "09_final.png", out)

    imwrite_unicode(args.output, out)
    print(f"[info] Wrote {args.output}")

    if args.show:
        try:
            cv2.imshow("input", image)
            cv2.imshow("scan", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"[warn] --show failed (no GUI?): {exc}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)