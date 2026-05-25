"""
ucode_dimensions.py
======================================

More accurate medicine length, breadth, and height measurement from top/side
images using:
- YOLO for automatic object detection
- SAM segmentation from the selected detection box
- 20 mm ArUco marker calibration
- Optional camera undistortion
- Marker-plane perspective rectification
- SAM mask scoring instead of blindly taking the highest SAM score
- Marker-aware filtering so the ArUco marker is not measured as the object


Marker:
  20 mm side, cv2.aruco.DICT_ARUCO_ORIGINAL

Optional camera calibration:
  Save camera_matrix and dist_coeffs in camera_calibration.npz:
    np.savez("camera_calibration.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)

Important capture rule:
  For top measurement, the marker and medicine must be on the same plane.
  For side/height measurement, the marker should be in the same vertical plane
  as the side face being measured. Otherwise any pixel scale will be biased.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

SEGMENT_ANYTHING_DIR = "/Users/paushali.mondal/Desktop/hackeasy/segment-anything"
if os.path.isdir(SEGMENT_ANYTHING_DIR) and SEGMENT_ANYTHING_DIR not in sys.path:
    sys.path.insert(0, SEGMENT_ANYTHING_DIR)

from segment_anything import sam_model_registry, SamPredictor


# ============================================================
# CONFIG
# ============================================================

TOP_IMAGE = "/Users/paushali.mondal/Desktop/ucode_dim_proj/images/img09.jpg"
SIDE_IMAGE = "/Users/paushali.mondal/Desktop/ucode_dim_proj/images/img11.jpg"

ASSET_DIR = Path("/Users/paushali.mondal/Desktop/hackeasy")
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
YOLO_MODEL = "yolo11n.pt"
YOLO_CONF = 0.15

CAMERA_CALIBRATION_FILE = "camera_calibration.npz"
USE_CAMERA_UNDISTORT = True

MARKER_SIDE_MM = 20.0
ARUCO_DICT_TYPE = cv2.aruco.DICT_ARUCO_ORIGINAL
ARUCO_MARKER_ID = None

# Rectification converts the detected marker plane into a virtual top-down
# metric image. Increase this for finer sub-mm contours, at the cost of memory.
USE_PERSPECTIVE_RECTIFICATION = True
RECTIFIED_PX_PER_MM = 12.0
RECTIFIED_BORDER_PX = 80
MAX_RECTIFIED_SIDE_PX = 5000
SHOW_PLOTS = False
SAVE_SUMMARY_PLOT = False

LOW_CONTRAST_THRESHOLD = 30
CANNY_LOW = 30
CANNY_HIGH = 120
LAPLACIAN_KSIZE = 3
UPSCALE_SMALL_IMAGES = True
TARGET_MIN_SIDE_PX = 900
TOP_TARGET_MODE = "label"  # "label", "packet", or "sam"
TOP_FULL_PACKET_BOX_EXPAND_RATIO = 0.08
TOP_PRIOR_REPLACE_COVERAGE = 0.82

MIN_COMPONENT_AREA_RATIO = 0.002
ARUCO_BOX_PADDING_PX = 10
ARUCO_REJECT_IOU = 0.01
ARUCO_REJECT_COVERAGE = 0.08

MEDICINE_CLASS_HINTS = {
    "pill",
    "tablet",
    "capsule",
    "caplet",
    "softgel",
    "medicine bottle",
    "pill bottle",
    "bottle",
    "vial",
    "jar",
    "sachet",
    "packet",
    "pouch",
    "blister pack",
    "strip",
    "box",
    "syrup",
    "dropper bottle",
    "spray bottle",
    "inhaler",
    "medicine",
    "medication",
    "drug",
}


def resolve_asset_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path)

    asset_path = ASSET_DIR / path
    if asset_path.exists():
        return str(asset_path)

    return str(path)


def image_quality_report(image_bgr, label):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())
    brightness = float(gray.mean())
    print(
        f"  [{label}] Quality: brightness={brightness:.1f}, "
        f"contrast={contrast:.1f}, blur={blur_score:.1f}"
    )
    if blur_score < 45:
        print(f"  [{label}] Warning: image is blurry; result will be less reliable.")
    if brightness < 80 or contrast < 35:
        print(f"  [{label}] Low light/contrast detected; preprocessing will be strengthened.")


def prepare_poor_quality_image(image_bgr, label):
    """Improve tiny, dark, or low-contrast captures before detection/segmentation."""
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()

    if UPSCALE_SMALL_IMAGES and min(h, w) < TARGET_MIN_SIDE_PX:
        scale = TARGET_MIN_SIDE_PX / float(min(h, w))
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(f"  [{label}] Upscaled image by {scale:.2f}x for marker/SAM stability.")

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    out = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 95:
        gamma = 0.65
        lut = np.array(
            [min(255, int((i / 255.0) ** gamma * 255.0)) for i in range(256)],
            dtype=np.uint8,
        )
        out = cv2.LUT(out, lut)

    denoised = cv2.fastNlMeansDenoisingColored(out, None, 4, 4, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.2)
    return cv2.addWeighted(denoised, 1.45, blurred, -0.45, 0)


# ============================================================
# OPTIONAL CAMERA UNDISTORTION
# ============================================================

def load_camera_calibration(path):
    if not USE_CAMERA_UNDISTORT or not os.path.exists(path):
        return None

    data = np.load(path)
    if "camera_matrix" not in data or "dist_coeffs" not in data:
        print(f"Camera calibration file found, but missing camera_matrix/dist_coeffs: {path}")
        return None

    print(f"Camera calibration loaded: {path}")
    return {
        "camera_matrix": data["camera_matrix"].astype(np.float64),
        "dist_coeffs": data["dist_coeffs"].astype(np.float64),
    }


CAMERA_CALIBRATION = load_camera_calibration(CAMERA_CALIBRATION_FILE)


def undistort_if_available(image_bgr):
    if CAMERA_CALIBRATION is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    camera_matrix = CAMERA_CALIBRATION["camera_matrix"]
    dist_coeffs = CAMERA_CALIBRATION["dist_coeffs"]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1.0, newImgSize=(w, h)
    )
    undistorted = cv2.undistort(image_bgr, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        cropped = undistorted[y:y + rh, x:x + rw]
        if cropped.size:
            return cropped
    return undistorted


# ============================================================
# ARUCO UTILITIES
# ============================================================

def enhance_for_aruco(gray):
    candidates = [("original", gray)]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    candidates.append(("CLAHE", clahe_img))

    clahe_s = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    candidates.append(("CLAHE-strong", clahe_s.apply(gray)))

    for gamma, lbl in [(0.5, "gamma-0.5"), (0.35, "gamma-0.35")]:
        lut = np.array(
            [min(255, int((i / 255) ** gamma * 255)) for i in range(256)],
            dtype=np.uint8,
        )
        candidates.append((lbl, cv2.LUT(gray, lut)))

    gamma_lut = np.array(
        [min(255, int((i / 255) ** 0.5 * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    candidates.append(("CLAHE+gamma", cv2.LUT(clahe_img, gamma_lut)))

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    candidates.append(("CLAHE+sharpen", cv2.filter2D(clahe_img, -1, sharpen_kernel)))

    adaptive = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4
    )
    candidates.append(("adaptive-thresh", adaptive))
    return candidates


def find_aruco_markers(image_bgr, quiet=False):
    """Return detected marker records with corners, id, bbox, area, and center."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    found_corners, found_ids, found_label = None, None, None

    for label, cand in enhance_for_aruco(gray):
        corners, ids, _ = detector.detectMarkers(cand)
        if ids is not None and len(ids) > 0:
            found_corners, found_ids, found_label = corners, ids, label
            break

    if found_ids is None:
        if not quiet:
            print("  ArUco marker not detected.")
        return []

    markers = []
    for idx in range(len(found_ids)):
        pts = found_corners[idx][0].astype(np.float32)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        markers.append({
            "id": int(found_ids[idx][0]),
            "corners": found_corners[idx],
            "bbox": np.array([x, y, x + w, y + h], dtype=float),
            "area": float(w * h),
            "center": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
            "enhancement": found_label,
        })

    if not quiet:
        ids = [m["id"] for m in markers]
        print(f"  ArUco detected via [{found_label}], ids={ids}")
    return markers


def select_marker(markers):
    if not markers:
        return None

    if ARUCO_MARKER_ID is not None:
        matches = [m for m in markers if m["id"] == ARUCO_MARKER_ID]
        if not matches:
            raise ValueError(
                f"ArUco ID {ARUCO_MARKER_ID} not found. "
                f"Detected: {[m['id'] for m in markers]}"
            )
        return matches[0]

    return max(markers, key=lambda m: m["area"])


def padded_box(box, image_shape, pad):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box.astype(float)
    return np.array([
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ], dtype=float)


def box_intersection_area(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def marker_overlap_stats(candidate_box, marker_box):
    inter = box_intersection_area(candidate_box, marker_box)
    cand_area = max(1.0, box_area(candidate_box))
    marker_area = max(1.0, box_area(marker_box))
    union = cand_area + marker_area - inter
    return inter / max(1.0, union), inter / cand_area, inter / marker_area


def candidate_hits_aruco(candidate_box, marker_boxes):
    for marker_box in marker_boxes:
        iou, candidate_coverage, marker_coverage = marker_overlap_stats(candidate_box, marker_box)
        if iou >= ARUCO_REJECT_IOU:
            return True, f"IoU={iou:.3f}"
        if candidate_coverage >= ARUCO_REJECT_COVERAGE:
            return True, f"candidate overlap={candidate_coverage:.1%}"
        if marker_coverage >= 0.35 and box_area(candidate_box) <= box_area(marker_box) * 4:
            return True, f"marker coverage={marker_coverage:.1%}"
    return False, ""


def marker_mm_per_pixel(marker):
    pts = marker["corners"][0]
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    marker_side_px = float(np.mean(sides))
    mm_per_pixel = MARKER_SIDE_MM / marker_side_px
    return marker_side_px, mm_per_pixel


def detect_aruco_scale(image):
    markers = find_aruco_markers(image, quiet=False)
    marker = select_marker(markers)
    if marker is None:
        raise ValueError(
            "ArUco marker not detected after all enhancements.\n"
            "  - Ensure marker is fully visible and unoccluded.\n"
            "  - Avoid glare/heavy shadows.\n"
            "  - Print at exactly 20 mm on paper.\n"
            "  - Verify ARUCO_DICT_TYPE is DICT_ARUCO_ORIGINAL."
        )

    marker_side_px, mm_per_pixel = marker_mm_per_pixel(marker)
    print(f"ArUco ID         : {marker['id']}")
    print(f"Marker side      : {marker_side_px:.2f} px")
    print(f"Scale            : {mm_per_pixel:.5f} mm/px")

    cv2.aruco.drawDetectedMarkers(
        image,
        [m["corners"] for m in markers],
        np.array([[m["id"]] for m in markers], dtype=np.int32),
    )
    return marker, mm_per_pixel


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO model...")
yolo_model_path = resolve_asset_path(YOLO_MODEL)
yolo_model = YOLO(yolo_model_path)
print(f"YOLO loaded  : {yolo_model_path}")

print("Loading SAM model...")
sam_checkpoint_path = resolve_asset_path(SAM_CHECKPOINT)
sam = sam_model_registry[MODEL_TYPE](checkpoint=sam_checkpoint_path)
predictor = SamPredictor(sam)
print(f"SAM loaded   : {sam_checkpoint_path}")


# ============================================================
# AUTO DETECTION
# ============================================================

def _pick_best_detection(detections, img_area, label, marker_boxes=None):
    """Filter marker candidates out and return the best medicine candidate."""
    marker_boxes = marker_boxes or []
    filtered = []

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        bw, bh = x2 - x1, y2 - y1
        squareness = min(bw, bh) / max(bw, bh + 1e-6)
        rel_area = d["area"] / img_area

        hits_marker, reason = candidate_hits_aruco(d["box"], marker_boxes)
        if hits_marker:
            print(f"  [{label}] Skip ArUco-overlap box: {d['cls']} ({reason})")
            continue

        if squareness > 0.75 and rel_area < 0.05:
            print(
                f"  [{label}] Skip ArUco-like square box: {d['cls']} "
                f"(sq={squareness:.2f} rel={rel_area:.3f})"
            )
            continue

        filtered.append(d)

    if not filtered:
        non_marker = []
        for d in detections:
            hits_marker, _ = candidate_hits_aruco(d["box"], marker_boxes)
            if not hits_marker:
                non_marker.append(d)
        filtered = non_marker

    if not filtered:
        raise ValueError(f"[{label}] Only ArUco marker candidates were detected.")

    hinted = [d for d in filtered if d["cls"] in MEDICINE_CLASS_HINTS]
    pool = hinted if hinted else filtered

    # Favor large, confident boxes, but still give a boost to medicine-like labels.
    def score(d):
        cls_boost = 1.25 if d["cls"] in MEDICINE_CLASS_HINTS else 1.0
        return d["area"] * (0.5 + d["conf"]) * cls_boost

    return max(pool, key=score)


def _yolo_detections(image_bgr, conf):
    results = yolo_model(image_bgr, conf=conf, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return []

    names = yolo_model.model.names
    out = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, "unknown").lower()
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append({
            "cls": cls_name,
            "conf": conf_val,
            "box": np.array([x1, y1, x2, y2], dtype=float),
            "area": float((x2 - x1) * (y2 - y1)),
        })
    return out


def _opencv_contour_fallback(image_bgr, label, marker_boxes=None):
    print(f"  [{label}] Level 3 - OpenCV contour fallback")
    marker_boxes = marker_boxes or []
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(
        blurred,
        max(10, int(otsu_val * 0.3)),
        max(30, int(otsu_val * 0.8)),
    )

    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    dilated = cv2.dilate(closed, np.ones((5, 5), np.uint8), iterations=2)

    for marker_box in marker_boxes:
        x1, y1, x2, y2 = padded_box(marker_box, image_bgr.shape, ARUCO_BOX_PADDING_PX).astype(int)
        dilated[y1:y2, x1:x2] = 0

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"[{label}] OpenCV contour fallback found no contours.")

    img_area = h * w
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.005 * img_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        candidate_box = np.array([x, y, x + bw, y + bh], dtype=float)
        squareness = min(bw, bh) / max(bw, bh + 1e-6)
        rel_area = area / img_area

        hits_marker, reason = candidate_hits_aruco(candidate_box, marker_boxes)
        if hits_marker:
            print(f"  [{label}] Skip contour overlapping ArUco ({reason})")
            continue

        if squareness > 0.75 and rel_area < 0.05:
            continue

        candidates.append({
            "cls": "contour",
            "conf": 1.0,
            "box": candidate_box,
            "area": float(bw * bh),
        })

    if not candidates:
        raise ValueError(f"[{label}] No valid contour candidates after marker filtering.")

    best = max(candidates, key=lambda d: d["area"])
    print(
        f"  [{label}] OpenCV contour selected: "
        f"box={best['box'].astype(int).tolist()} area={best['area']:.0f}"
    )
    return best["box"].astype(int), candidates


def _aruco_relative_fallback(image_bgr, label, markers):
    print(f"  [{label}] Level 4 - ArUco-relative positioning heuristic")
    if not markers:
        raise ValueError(f"[{label}] ArUco marker not found.")

    h, w = image_bgr.shape[:2]
    marker = max(markers, key=lambda m: m["area"])
    mx, my = marker["center"]

    pad = 10
    if mx < w / 2:
        x1 = int(mx + (w - mx) * 0.05)
        x2 = w - pad
    else:
        x1 = pad
        x2 = int(mx * 0.95)

    if label.lower() == "side":
        y1 = pad
        y2 = h - pad
    elif my < h / 2:
        y1 = int(my + (h - my) * 0.05)
        y2 = h - pad
    else:
        y1 = pad
        y2 = int(my * 0.95)

    box = np.array([x1, y1, x2, y2], dtype=int)
    print(f"  [{label}] ArUco @ ({mx:.0f},{my:.0f}) -> medicine box={box.tolist()}")
    detections = [{
        "cls": "aruco-heuristic",
        "conf": 1.0,
        "box": box.astype(float),
        "area": float((x2 - x1) * (y2 - y1)),
    }]
    return box, detections


def auto_detect_box(image_bgr, label="view"):
    h, w = image_bgr.shape[:2]
    img_area = h * w

    markers = find_aruco_markers(image_bgr, quiet=False)
    marker_boxes = [
        padded_box(m["bbox"], image_bgr.shape, ARUCO_BOX_PADDING_PX)
        for m in markers
    ]

    print(f"  [{label}] Level 1 - YOLO (conf={YOLO_CONF})")
    dets = _yolo_detections(image_bgr, YOLO_CONF)
    if dets:
        print(f"  [{label}] YOLO detections ({len(dets)}):")
        for d in dets:
            print(f"    {d['cls']:20s} conf={d['conf']:.2f} area={d['area']:.0f}")
        try:
            best = _pick_best_detection(dets, img_area, label, marker_boxes)
            print(f"  [{label}] Level 1 selected '{best['cls']}' box={best['box'].astype(int).tolist()}")
            return best["box"].astype(int), dets, "YOLO"
        except ValueError as exc:
            print(f"  [{label}] Level 1 rejected: {exc}")

    print(f"  [{label}] Level 2 - YOLO confidence sweep (0.08 -> 0.01)")
    for sweep_conf in [0.08, 0.05, 0.03, 0.01]:
        dets = _yolo_detections(image_bgr, sweep_conf)
        if not dets:
            continue
        print(f"  [{label}] Level 2 found {len(dets)} detection(s) at conf={sweep_conf}")
        for d in dets:
            print(f"    {d['cls']:20s} conf={d['conf']:.2f} area={d['area']:.0f}")
        try:
            best = _pick_best_detection(dets, img_area, label, marker_boxes)
            print(f"  [{label}] Level 2 selected '{best['cls']}' box={best['box'].astype(int).tolist()}")
            return best["box"].astype(int), dets, f"YOLO-sweep-{sweep_conf}"
        except ValueError as exc:
            print(f"  [{label}] Level 2 conf={sweep_conf} rejected: {exc}")

    try:
        box, dets = _opencv_contour_fallback(image_bgr, label, marker_boxes)
        return box, dets, "OpenCV-contour"
    except ValueError as exc:
        print(f"  [{label}] Level 3 failed: {exc}")

    try:
        box, dets = _aruco_relative_fallback(image_bgr, label, markers)
        return box, dets, "ArUco-heuristic"
    except ValueError as exc:
        print(f"  [{label}] Level 4 failed: {exc}")

    raise RuntimeError(
        f"[{label}] All detection levels failed. Check lighting, marker visibility, "
        "and object placement."
    )


def draw_all_detections(image_bgr, detections, chosen_box, label, method):
    vis = image_bgr.copy()

    markers = find_aruco_markers(image_bgr, quiet=True)
    for marker in markers:
        x1, y1, x2, y2 = marker["bbox"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            vis, f"ArUco {marker['id']}", (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
        )

    for d in detections:
        x1, y1, x2, y2 = d["box"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(
            vis, f"{d['cls']} {d['conf']:.2f}", (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

    cx1, cy1, cx2, cy2 = chosen_box
    cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 220, 255), 3)
    cv2.putText(
        vis, f"[{method}]", (cx1, max(cy1 - 8, 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2
    )
    return vis


# ============================================================
# EDGE / MASK UTILITIES
# ============================================================

def compute_edge_depth_map(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=LAPLACIAN_KSIZE)
    lap_abs = np.abs(lap)
    mean_sq = cv2.boxFilter(lap_abs ** 2, -1, (15, 15))
    mean_val = cv2.boxFilter(lap_abs, -1, (15, 15))
    variance = np.clip(mean_sq - mean_val ** 2, 0, None)
    depth = np.sqrt(variance).astype(np.float32)
    if depth.max() > 0:
        depth = depth / depth.max() * 255.0
    return depth.astype(np.uint8)


def background_contrast_score(image_bgr, box):
    x1, y1, x2, y2 = box.astype(int)
    h_img, w_img = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, True

    pad = 20
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = min(w_img, x2 + pad), min(h_img, y2 + pad)

    border_mask = np.zeros_like(gray)
    border_mask[by1:by2, bx1:bx2] = 255
    border_mask[y1:y2, x1:x2] = 0

    border_pixels = gray[border_mask == 255]
    if border_pixels.size == 0:
        return float(roi.std()), roi.std() < LOW_CONTRAST_THRESHOLD

    score = abs(float(roi.mean()) - float(border_pixels.mean()))
    is_low = score < LOW_CONTRAST_THRESHOLD
    print(f"  Contrast score (ROI vs border): {score:.1f} {'[LOW - enhancing]' if is_low else '[OK]'}")
    return score, is_low


def enhance_mask_with_edges(image_bgr, coarse_mask, box):
    x1, y1, x2, y2 = box.astype(int)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    depth_map = compute_edge_depth_map(gray)

    roi_gray = gray[y1:y2, x1:x2]
    if roi_gray.size == 0:
        return coarse_mask, depth_map

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi_gray)
    roi_bilateral = cv2.bilateralFilter(roi_enhanced, 9, 75, 75)

    otsu_thresh, _ = cv2.threshold(roi_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_lo = max(CANNY_LOW, int(otsu_thresh * 0.4))
    canny_hi = max(CANNY_HIGH, int(otsu_thresh * 0.9))
    edges_roi = cv2.Canny(roi_bilateral, canny_lo, canny_hi)

    edges_closed = cv2.morphologyEx(edges_roi, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours_roi, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_roi:
        print("  Edge enhancement: no contours - keeping SAM mask.")
        return coarse_mask, depth_map

    contours_full = [c + np.array([x1, y1]) for c in contours_roi]

    def edge_score(cnt):
        tmp = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, thickness=3)
        vals = depth_map[tmp == 255]
        return float(vals.mean()) if vals.size else 0.0

    scored = sorted(
        contours_full,
        key=lambda c: edge_score(c) * cv2.contourArea(c),
        reverse=True,
    )
    coarse_a = float(np.count_nonzero(coarse_mask))
    best_cnt = next((c for c in scored if cv2.contourArea(c) >= 0.30 * coarse_a), None)
    if best_cnt is None:
        print("  Edge enhancement: no large enough candidate - keeping SAM mask.")
        return coarse_mask, depth_map

    refined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(refined, [best_cnt], -1, 255, cv2.FILLED)

    blended = cv2.bitwise_or(coarse_mask, refined)
    blended = cv2.morphologyEx(blended, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    blended = cv2.morphologyEx(blended, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    overlap = np.count_nonzero(cv2.bitwise_and(coarse_mask, refined)) / max(1, np.count_nonzero(refined))
    print(f"  Edge enhancement: SAM/Canny overlap={overlap:.2%} edge-depth score={edge_score(best_cnt):.1f}")
    return blended, depth_map


def remove_aruco_from_mask(image_bgr, mask):
    """Prevent marker pixels from becoming the largest measured contour."""
    cleaned = mask.copy()
    markers = find_aruco_markers(image_bgr, quiet=True)
    for marker in markers:
        pts = marker["corners"][0].astype(np.int32)
        cv2.fillConvexPoly(cleaned, pts, 0)
        x1, y1, x2, y2 = padded_box(marker["bbox"], image_bgr.shape, ARUCO_BOX_PADDING_PX).astype(int)
        cleaned[y1:y2, x1:x2] = 0
    return cleaned


def clean_mask_components(mask, image_shape, box=None, min_area_ratio=MIN_COMPONENT_AREA_RATIO):
    """Keep plausible object components and remove tiny SAM/edge leftovers."""
    h, w = image_shape[:2]
    img_area = h * w
    binary = (mask > 0).astype(np.uint8) * 255

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    components = []
    box_center = None
    if box is not None:
        x1, y1, x2, y2 = box.astype(float)
        box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_ratio * img_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        comp_box = np.array([x, y, x + bw, y + bh], dtype=float)
        overlap = 0.0
        center_score = 0.0

        if box is not None:
            overlap = box_intersection_area(comp_box, box) / max(1.0, box_area(comp_box))
        if box_center is not None:
            center = np.array(centroids[i])
            max_dist = np.linalg.norm([w, h])
            center_score = 1.0 - min(1.0, np.linalg.norm(center - box_center) / max_dist)

        components.append((area * (1.0 + overlap + center_score), i))

    if not components:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = largest
    else:
        keep = max(components, key=lambda item: item[0])[1]

    cleaned = np.zeros_like(binary)
    cleaned[labels == keep] = 255
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cleaned


def build_side_edge_prior_mask(image_bgr, box):
    """Find the tall thin side face when SAM is confused by blur/shadows."""
    x1, y1, x2, y2 = box.astype(int)
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    roi = gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return None

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    enhanced = clahe.apply(roi)
    enhanced = cv2.bilateralFilter(enhanced, 7, 50, 50)

    otsu, _ = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(
        enhanced,
        max(10, int(otsu * 0.25)),
        max(35, int(otsu * 0.75)),
    )
    vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((21, 5), np.uint8), iterations=2)
    vertical = cv2.dilate(vertical, np.ones((7, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    marker_boxes = [
        padded_box(m["bbox"], image_bgr.shape, ARUCO_BOX_PADDING_PX)
        for m in find_aruco_markers(image_bgr, quiet=True)
    ]

    candidates = []
    roi_h, roi_w = roi.shape[:2]
    for cnt in contours:
        cnt_full = cnt + np.array([[[x1, y1]]])
        cx, cy, bw, bh = cv2.boundingRect(cnt_full)
        area = cv2.contourArea(cnt_full)
        if area < 0.0004 * h * w:
            continue
        if bh < 0.28 * h:
            continue
        if bw > 0.35 * roi_w or bw < 3:
            continue

        candidate_box = np.array([cx, cy, cx + bw, cy + bh], dtype=float)
        hits_marker, _ = candidate_hits_aruco(candidate_box, marker_boxes)
        if hits_marker:
            continue

        aspect_score = bh / max(1.0, bw)
        center_bonus = 1.0 - min(1.0, abs((cx + bw / 2.0) - ((x1 + x2) / 2.0)) / max(1.0, roi_w))
        candidates.append((area * aspect_score * (1.0 + center_bonus), cnt_full))

    if not candidates:
        return None

    best = max(candidates, key=lambda item: item[0])[1]
    prior = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(prior, [best], -1, 255, thickness=cv2.FILLED)
    prior = cv2.morphologyEx(prior, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8), iterations=2)
    prior = cv2.dilate(prior, np.ones((5, 3), np.uint8), iterations=1)
    return prior


def build_top_full_packet_prior_mask(image_bgr, box):
    """Estimate the complete top packet footprint, not just the printed label."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(float)
    bw, bh = x2 - x1, y2 - y1
    pad = TOP_FULL_PACKET_BOX_EXPAND_RATIO * max(bw, bh)
    ex1 = int(max(0, x1 - pad))
    ey1 = int(max(0, y1 - pad))
    ex2 = int(min(w, x2 + pad))
    ey2 = int(min(h, y2 + pad))

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    roi = image_bgr[ey1:ey2, ex1:ex2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(
        blur,
        max(10, int(otsu_val * 0.25)),
        max(35, int(otsu_val * 0.85)),
    )
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=2)
    edges = cv2.dilate(edges, np.ones((9, 9), np.uint8), iterations=2)

    marker_boxes = [
        padded_box(m["bbox"], image_bgr.shape, ARUCO_BOX_PADDING_PX)
        for m in find_aruco_markers(image_bgr, quiet=True)
    ]
    for marker_box in marker_boxes:
        mx1, my1, mx2, my2 = marker_box.astype(int)
        mx1 = max(0, mx1 - ex1)
        mx2 = min(edges.shape[1], mx2 - ex1)
        my1 = max(0, my1 - ey1)
        my2 = min(edges.shape[0], my2 - ey1)
        if mx2 > mx1 and my2 > my1:
            edges[my1:my2, mx1:mx2] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prior = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    candidate_contours = []
    min_area = 0.20 * max(1.0, bw * bh)

    for cnt in contours:
        cnt_full = cnt + np.array([[[ex1, ey1]]])
        area = cv2.contourArea(cnt_full)
        if area < min_area:
            continue

        candidate_box = np.array(cv2.boundingRect(cnt_full), dtype=float)
        candidate_box = np.array([
            candidate_box[0],
            candidate_box[1],
            candidate_box[0] + candidate_box[2],
            candidate_box[1] + candidate_box[3],
        ])
        box_overlap = box_intersection_area(candidate_box, box) / max(1.0, box_area(candidate_box))
        hits_marker, _ = candidate_hits_aruco(candidate_box, marker_boxes)
        if box_overlap < 0.45 or hits_marker:
            continue

        candidate_contours.append((area * (1.0 + box_overlap), cnt_full))

    if candidate_contours:
        best = max(candidate_contours, key=lambda item: item[0])[1]
        hull = cv2.convexHull(best)
        cv2.drawContours(prior, [hull], -1, 255, thickness=cv2.FILLED)
    else:
        # Fallback to the detector box; for sachets this is usually closer to
        # the packet footprint than SAM's inner printed-label mask.
        cv2.rectangle(prior, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=cv2.FILLED)

    prior = remove_aruco_from_mask(image_bgr, prior)
    prior = cv2.morphologyEx(prior, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)
    return clean_mask_components(prior, image_bgr.shape, box, min_area_ratio=0.001)


def build_top_label_prior_mask(image_bgr, box):
    """Estimate the complete printed label panel inside the detected packet."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Printed medicine labels are usually a large low-saturation, brighter
    # panel surrounded by darker foil/cardboard. This catches the full panel,
    # while the morphology reconnects text gaps and logo holes.
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    lightness = lab[:, :, 0]
    label_mask = np.zeros(gray.shape, dtype=np.uint8)
    label_mask[((sat < 95) & (val > 80)) | ((sat < 125) & (lightness > 115))] = 255

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    enhanced = clahe.apply(gray)
    _, otsu_light = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    label_mask = cv2.bitwise_or(label_mask, otsu_light)

    label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=2)
    label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)

    marker_boxes = [
        padded_box(m["bbox"], image_bgr.shape, ARUCO_BOX_PADDING_PX)
        for m in find_aruco_markers(image_bgr, quiet=True)
    ]
    for marker_box in marker_boxes:
        mx1, my1, mx2, my2 = marker_box.astype(int)
        mx1 = max(0, mx1 - x1)
        mx2 = min(label_mask.shape[1], mx2 - x1)
        my1 = max(0, my1 - y1)
        my2 = min(label_mask.shape[0], my2 - y1)
        if mx2 > mx1 and my2 > my1:
            label_mask[my1:my2, mx1:mx2] = 0

    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        prior = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.rectangle(prior, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
        prior = remove_aruco_from_mask(image_bgr, prior)
        print("  [top] Label-area prior using detector box fallback.")
        return clean_mask_components(prior, image_bgr.shape, box, min_area_ratio=0.0008)

    roi_area = max(1.0, float((x2 - x1) * (y2 - y1)))
    roi_center = np.array([(x2 - x1) / 2.0, (y2 - y1) / 2.0])
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.08 * roi_area or area > 0.92 * roi_area:
            continue

        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if rw < 0.25 * (x2 - x1) or rh < 0.20 * (y2 - y1):
            continue

        rect = cv2.minAreaRect(cnt)
        rect_area = max(1.0, rect[1][0] * rect[1][1])
        fill = area / rect_area
        if fill < 0.35:
            continue

        center = np.array([rx + rw / 2.0, ry + rh / 2.0])
        dist_score = 1.0 - min(1.0, np.linalg.norm(center - roi_center) / np.linalg.norm(roi_center))
        candidates.append((area * (0.75 + fill + dist_score), cnt))

    if not candidates:
        prior = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.rectangle(prior, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
        prior = remove_aruco_from_mask(image_bgr, prior)
        print("  [top] Label-area prior using detector box fallback.")
        return clean_mask_components(prior, image_bgr.shape, box, min_area_ratio=0.0008)

    best = max(candidates, key=lambda item: item[0])[1]
    hull = cv2.convexHull(best)
    prior_roi = np.zeros(label_mask.shape, dtype=np.uint8)
    cv2.drawContours(prior_roi, [hull], -1, 255, thickness=cv2.FILLED)
    prior_roi = cv2.morphologyEx(prior_roi, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)

    prior = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    prior[y1:y2, x1:x2] = prior_roi
    prior = remove_aruco_from_mask(image_bgr, prior)
    return clean_mask_components(prior, image_bgr.shape, box, min_area_ratio=0.0008)


def refine_top_mask(image_bgr, mask, box):
    if TOP_TARGET_MODE == "sam":
        print("  [top] Keeping SAM mask because TOP_TARGET_MODE='sam'.")
        return mask

    if TOP_TARGET_MODE == "label":
        prior = build_top_label_prior_mask(image_bgr, box)
        prior_name = "label-area"
    else:
        prior = build_top_full_packet_prior_mask(image_bgr, box)
        prior_name = "full-packet"

    if prior is None or np.count_nonzero(prior) == 0:
        print(f"  [top] {prior_name} prior: no stable candidate.")
        return mask

    sam_area = np.count_nonzero(mask)
    prior_area = np.count_nonzero(prior)
    coverage = np.count_nonzero(cv2.bitwise_and(mask, prior)) / max(1, prior_area)
    area_ratio = sam_area / max(1, prior_area)

    if coverage < TOP_PRIOR_REPLACE_COVERAGE or area_ratio < TOP_PRIOR_REPLACE_COVERAGE:
        print(
            f"  [top] Replacing SAM mask with {prior_name} prior "
            f"(coverage={coverage:.1%}, area_ratio={area_ratio:.1%})."
        )
        return prior

    print(f"  [top] Blending SAM with {prior_name} prior (coverage={coverage:.1%}).")
    blended = cv2.bitwise_or(mask, prior)
    return clean_mask_components(blended, image_bgr.shape, box, min_area_ratio=0.001)


def refine_side_mask(image_bgr, mask, box):
    prior = build_side_edge_prior_mask(image_bgr, box)
    if prior is None:
        print("  [side] Side edge prior: no stable thin-face candidate.")
        return mask

    overlap = np.count_nonzero(cv2.bitwise_and(mask, prior)) / max(1, np.count_nonzero(prior))
    mask_contour = largest_contour(mask)
    _, major_px, minor_px = measure_object_px(mask_contour)
    aspect = major_px / max(1.0, minor_px)

    if overlap < 0.25 or aspect < 2.0:
        print(
            f"  [side] Replacing SAM mask with side edge prior "
            f"(overlap={overlap:.1%}, aspect={aspect:.2f})."
        )
        return clean_mask_components(prior, image_bgr.shape, box, min_area_ratio=0.0005)

    print(f"  [side] Blending side edge prior (overlap={overlap:.1%}, aspect={aspect:.2f}).")
    blended = cv2.bitwise_or(mask, prior)
    return clean_mask_components(blended, image_bgr.shape, box, min_area_ratio=0.0005)


def visualize_edge_depth(image_bgr, depth_map, mask, title="Edge Depth"):
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(image_bgr, 0.55, depth_color, 0.45, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 128), 2)
    cv2.putText(overlay, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return overlay


# ============================================================
# SAM / GEOMETRY
# ============================================================

def sam_segment(image_bgr, box):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)

    x1, y1, x2, y2 = box.astype(int)
    box_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 255
    box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    img_area = image_bgr.shape[0] * image_bgr.shape[1]

    markers = find_aruco_markers(image_bgr, quiet=True)
    marker_masks = []
    for marker in markers:
        marker_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(marker_mask, marker["corners"][0].astype(np.int32), 255)
        marker_masks.append(marker_mask)

    ranked = []
    for idx, mask_bool in enumerate(masks):
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        area = float(np.count_nonzero(mask_uint8))
        if area < 1:
            continue

        box_overlap = np.count_nonzero(cv2.bitwise_and(mask_uint8, box_mask)) / area
        marker_overlap = 0.0
        for marker_mask in marker_masks:
            marker_overlap = max(
                marker_overlap,
                np.count_nonzero(cv2.bitwise_and(mask_uint8, marker_mask)) / area,
            )

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            moments = cv2.moments(cnt)
            if moments["m00"] > 0:
                center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
            else:
                center = box_center
            contour_area = cv2.contourArea(cnt)
        else:
            center = box_center
            contour_area = area

        center_dist = np.linalg.norm(center - box_center)
        box_diag = np.linalg.norm([max(1, x2 - x1), max(1, y2 - y1)])
        center_score = 1.0 - min(1.0, center_dist / max(1.0, box_diag))
        area_ratio = area / img_area
        contour_fill = contour_area / max(1.0, area)

        score = (
            float(scores[idx]) * 2.0
            + box_overlap * 2.0
            + center_score
            + min(area_ratio * 20.0, 1.0)
            + contour_fill * 0.5
            - marker_overlap * 5.0
        )
        ranked.append((score, idx, box_overlap, marker_overlap, area))

    if not ranked:
        raise ValueError("SAM produced no valid masks.")

    ranked.sort(reverse=True, key=lambda item: item[0])
    best_score, best_idx, box_overlap, marker_overlap, area = ranked[0]
    print(
        f"  SAM selected mask {best_idx}: score={best_score:.2f}, "
        f"box_overlap={box_overlap:.1%}, marker_overlap={marker_overlap:.1%}, area={area:.0f}"
    )

    best_mask = masks[best_idx]
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    return cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def full_segment_pipeline(image_bgr, box, label="view"):
    print(f"\n  [{label}] Running SAM segmentation...")
    coarse_mask = sam_segment(image_bgr, box)
    coarse_mask = remove_aruco_from_mask(image_bgr, coarse_mask)
    coarse_mask = clean_mask_components(coarse_mask, image_bgr.shape, box)

    _, is_low = background_contrast_score(image_bgr, box)
    if is_low:
        print(f"  [{label}] LOW CONTRAST - activating edge-depth enhancement.")
        final_mask, depth_map = enhance_mask_with_edges(image_bgr, coarse_mask, box)
        final_mask = remove_aruco_from_mask(image_bgr, final_mask)
        final_mask = clean_mask_components(final_mask, image_bgr.shape, box)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        depth_map = compute_edge_depth_map(gray)
        final_mask = coarse_mask

    if label.lower() == "top":
        final_mask = refine_top_mask(image_bgr, final_mask, box)
        final_mask = remove_aruco_from_mask(image_bgr, final_mask)
        final_mask = clean_mask_components(final_mask, image_bgr.shape, box, min_area_ratio=0.001)

    if label.lower() == "side":
        final_mask = refine_side_mask(image_bgr, final_mask, box)
        final_mask = remove_aruco_from_mask(image_bgr, final_mask)
        final_mask = clean_mask_components(final_mask, image_bgr.shape, box, min_area_ratio=0.0005)

    depth_overlay = visualize_edge_depth(
        image_bgr.copy(), depth_map, final_mask, title=f"Edge Depth [{label}]"
    )
    return final_mask, depth_overlay


def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in mask.")
    return max(contours, key=cv2.contourArea)


def measure_object_px(contour):
    rect = cv2.minAreaRect(contour)
    width_px, ht_px = rect[1]
    return rect, max(width_px, ht_px), min(width_px, ht_px)


def draw_measurement(image, rect, label):
    box = np.int32(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    cv2.putText(image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def resize_for_summary(img, max_side=1000):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def rectify_mask_to_marker_plane(image_bgr, mask, marker):
    """Warp the full mask into marker-plane coordinates.

    The destination coordinate system is metric: RECTIFIED_PX_PER_MM pixels
    represent 1 mm. This corrects perspective only when the measured object
    lies on the same plane as the marker.
    """
    marker_px = MARKER_SIDE_MM * RECTIFIED_PX_PER_MM
    src_marker = marker["corners"][0].astype(np.float32)
    dst_marker = np.array([
        [0, 0],
        [marker_px, 0],
        [marker_px, marker_px],
        [0, marker_px],
    ], dtype=np.float32)

    h, w = image_bgr.shape[:2]
    h_img_to_marker = cv2.getPerspectiveTransform(src_marker, dst_marker)
    image_corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(image_corners, h_img_to_marker).reshape(-1, 2)

    min_xy = np.floor(warped_corners.min(axis=0) - RECTIFIED_BORDER_PX)
    max_xy = np.ceil(warped_corners.max(axis=0) + RECTIFIED_BORDER_PX)
    out_w = int(max_xy[0] - min_xy[0])
    out_h = int(max_xy[1] - min_xy[1])

    if out_w <= 0 or out_h <= 0:
        raise ValueError("Invalid rectified canvas size.")

    scale_down = max(out_w / MAX_RECTIFIED_SIDE_PX, out_h / MAX_RECTIFIED_SIDE_PX, 1.0)
    effective_px_per_mm = RECTIFIED_PX_PER_MM / scale_down

    translate = np.array([
        [1.0 / scale_down, 0, -min_xy[0] / scale_down],
        [0, 1.0 / scale_down, -min_xy[1] / scale_down],
        [0, 0, 1],
    ], dtype=np.float64)

    h_total = translate @ h_img_to_marker
    out_size = (int(out_w / scale_down), int(out_h / scale_down))

    warped_mask = cv2.warpPerspective(mask, h_total, out_size, flags=cv2.INTER_NEAREST)
    warped_img = cv2.warpPerspective(image_bgr, h_total, out_size, flags=cv2.INTER_LINEAR)

    warped_mask = (warped_mask > 0).astype(np.uint8) * 255
    warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return warped_img, warped_mask, 1.0 / effective_px_per_mm


def measure_with_best_available_geometry(image_bgr, mask, label):
    markers = find_aruco_markers(image_bgr, quiet=False)
    marker = select_marker(markers)
    if marker is None:
        raise ValueError(f"[{label}] ArUco marker not found for measurement.")

    can_rectify = USE_PERSPECTIVE_RECTIFICATION and label.lower() != "side"
    if label.lower() == "side" and USE_PERSPECTIVE_RECTIFICATION:
        print("  [side] Skipping perspective rectification; scalar scale is safer for side thickness.")

    if can_rectify:
        try:
            warped_img, warped_mask, mm_per_px = rectify_mask_to_marker_plane(image_bgr, mask, marker)
            warped_mask = clean_mask_components(warped_mask, warped_img.shape)
            contour = largest_contour(warped_mask)
            rect, dim1_px, dim2_px = measure_object_px(contour)
            print(f"  [{label}] Measurement geometry: marker-plane rectified")
            print(f"  [{label}] Rectified scale: {mm_per_px:.5f} mm/px")
            return {
                "rect": rect,
                "dim1_mm": dim1_px * mm_per_px,
                "dim2_mm": dim2_px * mm_per_px,
                "mm_per_px": mm_per_px,
                "measurement_img": warped_img,
                "measurement_mask": warped_mask,
                "rectified": True,
            }
        except Exception as exc:
            print(f"  [{label}] Rectified measurement failed, using scalar marker scale: {exc}")

    marker_side_px, mm_per_px = marker_mm_per_pixel(marker)
    contour = largest_contour(mask)
    rect, dim1_px, dim2_px = measure_object_px(contour)
    print(f"  [{label}] Measurement geometry: scalar marker scale")
    print(f"  [{label}] Marker side: {marker_side_px:.2f} px, scale={mm_per_px:.5f} mm/px")
    return {
        "rect": rect,
        "dim1_mm": dim1_px * mm_per_px,
        "dim2_mm": dim2_px * mm_per_px,
        "mm_per_px": mm_per_px,
        "measurement_img": image_bgr,
        "measurement_mask": mask,
        "rectified": False,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_view(image_path, label):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"{image_path} not found.")

    img = undistort_if_available(img)
    image_quality_report(img, label)
    img = prepare_poor_quality_image(img, label)

    print(f"\n{'=' * 50}")
    print(f"{label.upper()} VIEW - {image_path}")
    print("=" * 50)

    print(f"\n[{label}] Step 1/4 - Auto-detection")
    box, all_detections, method = auto_detect_box(img, label=label)
    print(f"  [{label}] Detection method used: {method}")
    yolo_vis = draw_all_detections(img, all_detections, box, label=label, method=method)

    print(f"\n[{label}] Step 2/4 - SAM segmentation + edge-depth analysis")
    mask, depth_overlay = full_segment_pipeline(img, box, label=label)

    print(f"\n[{label}] Step 3/4 - Geometry measurement")
    measurement = measure_with_best_available_geometry(img, mask, label)

    print(f"\n[{label}] Step 4/4 - ArUco calibration visualization")
    marker_img = img.copy()
    detect_aruco_scale(marker_img)

    return {
        "img": img,
        "marker_img": marker_img,
        "yolo_vis": yolo_vis,
        "mask": mask,
        "depth_overlay": depth_overlay,
        "rect": measurement["rect"],
        "dim1_mm": measurement["dim1_mm"],
        "dim2_mm": measurement["dim2_mm"],
        "mm_per_px": measurement["mm_per_px"],
        "method": method,
        "measurement_img": measurement["measurement_img"],
        "measurement_mask": measurement["measurement_mask"],
        "rectified": measurement["rectified"],
    }


def draw_final_measurement(view_result, label_text):
    target = view_result["measurement_img"]
    draw_measurement(target, view_result["rect"], label_text)
    return target


def main():
    top = process_view(TOP_IMAGE, "top")
    length_mm = top["dim1_mm"]
    breadth_mm = top["dim2_mm"]
    top_final = draw_final_measurement(top, f"L={length_mm:.2f}mm  B={breadth_mm:.2f}mm")

    side = process_view(SIDE_IMAGE, "side")
    height_mm = side["dim2_mm"]
    side_final = draw_final_measurement(side, f"H={height_mm:.2f}mm")

    print("\n" + "=" * 40)
    print("FINAL MEASUREMENTS")
    print("=" * 40)
    print(f"  Length  : {length_mm:.2f} mm")
    print(f"  Breadth : {breadth_mm:.2f} mm")
    print(f"  Height  : {height_mm:.2f} mm")
    print("=" * 40)

    print("\nAccuracy notes:")
    print("  - Best accuracy requires the marker and measured face to be in the same plane.")
    print("  - Add camera_calibration.npz to reduce lens distortion error.")
    print("  - If rectified output looks stretched, disable USE_PERSPECTIVE_RECTIFICATION.")



if __name__ == "__main__":
    main()
