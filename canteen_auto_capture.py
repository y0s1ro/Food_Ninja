"""
Canteen food-waste prototype

Features:
- Automatic plate capture from USB camera (button-armed)
- Daily menu constraints (known categories for the day)
- FoodSAM inference call for each captured plate
- Per-category waste metrics and SQLite persistence

Run example:
python canteen_auto_capture.py --camera-id 0 --menu-config configs/daily_menu.sample.json --show-preview
"""

import argparse
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import deque

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CATEGORY_FILE = os.path.join(
    SCRIPT_DIR,
    "FoodSAM",
    "FoodSAM_tools",
    "category_id_files",
    "foodseg103_category_id.txt",
)
DEFAULT_SEMANTIC_SCRIPT = os.path.join(SCRIPT_DIR, "FoodSAM", "semantic.py")


def load_category_maps(category_file):
    id_to_name = {}
    name_to_id = {}
    with open(category_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            class_id = int(parts[0])
            class_name = parts[1].strip()
            id_to_name[class_id] = class_name
            name_to_id[class_name.lower()] = class_id
    return id_to_name, name_to_id


def resolve_menu_categories(menu_items, name_to_id):
    allowed_ids = set()
    unresolved = []

    for item in menu_items:
        if isinstance(item, int):
            allowed_ids.add(item)
            continue

        if isinstance(item, str):
            stripped = item.strip()
            if not stripped:
                continue
            if stripped.isdigit():
                allowed_ids.add(int(stripped))
                continue
            class_id = name_to_id.get(stripped.lower())
            if class_id is not None:
                allowed_ids.add(class_id)
            else:
                unresolved.append(stripped)
            continue

        unresolved.append(str(item))

    return allowed_ids, unresolved


def load_daily_menu(menu_config_path, menu_date, name_to_id):
    with open(menu_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    by_date = config.get("by_date", {})
    default_menu = config.get("default", [])
    menu_items = by_date.get(menu_date, default_menu)

    if not isinstance(menu_items, list):
        raise ValueError("Menu config must provide a list for each date/default")

    allowed_ids, unresolved = resolve_menu_categories(menu_items, name_to_id)
    allowed_ids.discard(0)
    return allowed_ids, unresolved, menu_items


def detect_plate_circle(frame, min_radius):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    max_radius = min(frame.shape[0], frame.shape[1]) // 2
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(80, frame.shape[0] // 3),
        param1=120,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    h, w = frame.shape[:2]
    cx0, cy0 = w // 2, h // 2

    # Prefer circles close to frame center to reduce false triggers.
    best = None
    best_score = None
    for x, y, r in circles:
        dist = np.hypot(x - cx0, y - cy0)
        score = dist - 0.4 * r
        if best_score is None or score < best_score:
            best = (x, y, r)
            best_score = score

    return best


def is_stable(window, center_tol=12, radius_tol=10):
    if not window:
        return False
    arr = np.array(window, dtype=np.float32)
    x_range = float(arr[:, 0].max() - arr[:, 0].min())
    y_range = float(arr[:, 1].max() - arr[:, 1].min())
    r_range = float(arr[:, 2].max() - arr[:, 2].min())
    return x_range <= center_tol and y_range <= center_tol and r_range <= radius_tol


def crop_plate_roi(frame, circle, margin_ratio=0.2):
    x, y, r = circle
    margin = int(r * margin_ratio)
    radius = r + margin

    h, w = frame.shape[:2]
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius)
    y2 = min(h, y + radius)

    crop = frame[y1:y2, x1:x2].copy()
    crop_circle = (x - x1, y - y1, r)
    bbox = (x1, y1, x2, y2)
    return crop, crop_circle, bbox


def run_foodsam_on_image(image_path, output_root, python_exec, semantic_script):
    cmd = [
        python_exec,
        semantic_script,
        "--img_path",
        image_path,
        "--output",
        output_root,
    ]

    proc = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "FoodSAM failed with code {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                proc.returncode,
                proc.stdout[-2000:],
                proc.stderr[-2000:],
            )
        )

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    run_dir = os.path.join(output_root, base_name)
    return run_dir


def choose_mask_path(run_dir):
    enhanced = os.path.join(run_dir, "enhance_mask.png")
    if os.path.exists(enhanced):
        return enhanced
    pred = os.path.join(run_dir, "pred_mask.png")
    if os.path.exists(pred):
        return pred
    raise FileNotFoundError("No enhance_mask.png or pred_mask.png found in {}".format(run_dir))


def log_debug(enabled, message):
    if enabled:
        print("[DEBUG] {}".format(message))


def compute_waste(mask_path, id_to_name, allowed_ids, roi_circle=None):
    mask_img = cv2.imread(mask_path)
    if mask_img is None:
        raise FileNotFoundError("Cannot read mask: {}".format(mask_path))

    mask = mask_img[:, :, 2].astype(np.int32)
    h, w = mask.shape

    roi = np.ones((h, w), dtype=bool)
    if roi_circle is not None:
        cx, cy, r = roi_circle
        yy, xx = np.ogrid[:h, :w]
        roi = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r ** 2)

    frame_pixels = int(h * w)
    roi_pixels = int(roi.sum())
    if roi_pixels <= 0:
        raise ValueError("ROI has zero pixels")

    values, counts = np.unique(mask[roi], return_counts=True)
    allowed_rows = []
    unexpected_pixels = 0

    for class_id, pixel_count in zip(values.tolist(), counts.tolist()):
        if class_id == 0:
            continue
        class_name = id_to_name.get(class_id, "class_{}".format(class_id))
        if allowed_ids and class_id not in allowed_ids:
            unexpected_pixels += int(pixel_count)
            continue

        pct = (float(pixel_count) / float(roi_pixels)) * 100.0
        allowed_rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "pixels": int(pixel_count),
                "pct": round(pct, 4),
            }
        )

    allowed_rows.sort(key=lambda x: x["pct"], reverse=True)
    total_food_pct = round(sum(row["pct"] for row in allowed_rows), 4)
    unexpected_pct = round((unexpected_pixels / float(roi_pixels)) * 100.0, 4)

    return {
        "rows": allowed_rows,
        "frame_pixels": frame_pixels,
        "roi_pixels": roi_pixels,
        "roi_pct_of_frame": round((roi_pixels / float(frame_pixels)) * 100.0, 4),
        "total_food_pct": total_food_pct,
        "unexpected_pct": unexpected_pct,
    }


def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            station_id TEXT,
            menu_date TEXT NOT NULL,
            menu_items_json TEXT NOT NULL,
            image_path TEXT NOT NULL,
            crop_path TEXT NOT NULL,
            run_dir TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            roi_pixels INTEGER NOT NULL,
            total_food_pct REAL NOT NULL,
            unexpected_pct REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS capture_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capture_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            pixels INTEGER NOT NULL,
            pct REAL NOT NULL,
            FOREIGN KEY (capture_id) REFERENCES captures(id)
        )
        """
    )
    conn.commit()
    return conn


def save_capture(conn, payload):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO captures (
            captured_at, station_id, menu_date, menu_items_json,
            image_path, crop_path, run_dir, mask_path,
            roi_pixels, total_food_pct, unexpected_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["captured_at"],
            payload["station_id"],
            payload["menu_date"],
            json.dumps(payload["menu_items"]),
            payload["image_path"],
            payload["crop_path"],
            payload["run_dir"],
            payload["mask_path"],
            payload["roi_pixels"],
            payload["total_food_pct"],
            payload["unexpected_pct"],
        ),
    )
    capture_id = cur.lastrowid

    rows = payload["rows"]
    cur.executemany(
        """
        INSERT INTO capture_items (capture_id, class_id, class_name, pixels, pct)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                capture_id,
                row["class_id"],
                row["class_name"],
                row["pixels"],
                row["pct"],
            )
            for row in rows
        ],
    )

    conn.commit()
    return capture_id


def print_summary(capture_id, stats):
    print("\n" + "=" * 66)
    print("Capture saved. ID: {}".format(capture_id))
    print("Total food coverage: {:.2f}%".format(stats["total_food_pct"]))
    print("Unexpected category coverage: {:.2f}%".format(stats["unexpected_pct"]))
    print("-" * 66)
    print("{:<4} {:<28} {:>10} {:>10}".format("ID", "Category", "Pixels", "Pct"))
    print("-" * 66)
    for row in stats["rows"]:
        print(
            "{:<4} {:<28} {:>10} {:>9.2f}%".format(
                row["class_id"], row["class_name"], row["pixels"], row["pct"]
            )
        )


def draw_preview(frame, circle, ready_for_capture, menu_text, arm_key):
    canvas = frame.copy()
    if circle is not None:
        x, y, r = circle
        cv2.circle(canvas, (x, y), r, (0, 220, 0), 2)
        cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)

    status = "ARMED" if ready_for_capture else "DISARMED"
    color = (0, 220, 0) if ready_for_capture else (0, 165, 255)
    cv2.putText(canvas, "Status: {}".format(status), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(canvas, menu_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    circle_status = "YES" if circle is not None else "NO"
    cv2.putText(canvas, "Plate circle detected: {}".format(circle_status), (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, "Press {} to arm capture".format(arm_key.upper()), (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, "Press q to quit", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Automatic canteen plate capture and waste analysis")
    parser.add_argument("--camera-id", type=int, default=0, help="USB camera index")
    parser.add_argument(
        "--menu-config",
        default=os.path.join("configs", "daily_menu.json"),
        help="JSON config with default and by_date menu categories",
    )
    parser.add_argument(
        "--category-file",
        default=DEFAULT_CATEGORY_FILE,
        help="FoodSeg category mapping file",
    )
    parser.add_argument(
        "--semantic-script",
        default=DEFAULT_SEMANTIC_SCRIPT,
        help="Path to FoodSAM semantic.py",
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python executable used to run FoodSAM semantic.py",
    )
    parser.add_argument("--output-dir", default=os.path.join("output", "canteen_runs"), help="Inference output dir")
    parser.add_argument("--captures-dir", default=os.path.join("output", "canteen_captures"), help="Saved capture images")
    parser.add_argument("--db-path", default=os.path.join("output", "canteen", "canteen_waste.db"), help="SQLite path")
    parser.add_argument("--station-id", default="station-1", help="Capture station identifier")
    parser.add_argument("--stability-frames", type=int, default=10, help="Frames required for stable plate")
    parser.add_argument("--min-plate-radius", type=int, default=100, help="Minimum detected plate radius in pixels")
    parser.add_argument("--cooldown-sec", type=float, default=2.0, help="Minimum seconds between captures")
    parser.add_argument("--arm-key", default="a", help="Keyboard key to arm a single auto-capture")
    parser.add_argument("--start-armed", action="store_true", help="Start in ARMED mode without key press")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--debug-every-n-frames", type=int, default=30, help="Emit periodic debug status every N frames")
    parser.add_argument("--show-preview", action="store_true", help="Show live camera preview")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.show_preview and not args.start_armed:
        raise ValueError("Use --show-preview for button arming or pass --start-armed to run headless")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.captures_dir, exist_ok=True)

    id_to_name, name_to_id = load_category_maps(args.category_file)

    menu_date = dt.date.today().isoformat()
    allowed_ids, unresolved, menu_items = load_daily_menu(args.menu_config, menu_date, name_to_id)

    if unresolved:
        print("Warning: unresolved menu items: {}".format(", ".join(unresolved)))
    if not allowed_ids:
        print("Warning: no allowed category IDs resolved; all non-background classes will be accepted")

    menu_text = "Menu categories: {}".format(
        ", ".join(str(x) for x in sorted(allowed_ids)) if allowed_ids else "ALL"
    )

    conn = init_db(args.db_path)
    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera {}".format(args.camera_id))

    stable_window = deque(maxlen=max(3, args.stability_frames))
    ready_for_capture = args.start_armed
    last_capture_ts = 0.0
    frame_idx = 0
    last_circle_state = None

    print("Auto-capture started. Date menu: {}".format(menu_date))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera")
                continue

            frame_idx += 1

            circle = detect_plate_circle(frame, args.min_plate_radius)
            circle_exists = circle is not None

            if last_circle_state is None or circle_exists != last_circle_state:
                if circle_exists:
                    x, y, r = circle
                    log_debug(args.debug, "Plate circle detected at x={}, y={}, r={}".format(x, y, r))
                else:
                    log_debug(args.debug, "Plate circle lost")
                last_circle_state = circle_exists

            if args.debug and args.debug_every_n_frames > 0 and frame_idx % args.debug_every_n_frames == 0:
                stable_count = len(stable_window)
                log_debug(
                    True,
                    "Frame {} status: circle_exists={}, armed={}, stable_window={}/{}".format(
                        frame_idx,
                        circle_exists,
                        ready_for_capture,
                        stable_count,
                        args.stability_frames,
                    ),
                )

            if circle is not None:
                stable_window.append(circle)

                can_capture = (
                    ready_for_capture
                    and len(stable_window) >= args.stability_frames
                    and is_stable(stable_window)
                    and (time.time() - last_capture_ts) >= args.cooldown_sec
                )

                if can_capture:
                    capture_ts = dt.datetime.now()
                    stamp = capture_ts.strftime("%Y%m%d_%H%M%S")
                    token = uuid.uuid4().hex[:8]
                    base_name = "plate_{}_{}".format(stamp, token)

                    image_path = os.path.join(args.captures_dir, base_name + "_full.jpg")
                    crop_path = os.path.join(args.captures_dir, base_name + "_crop.jpg")

                    cv2.imwrite(image_path, frame)

                    crop_img, crop_circle, _ = crop_plate_roi(frame, circle)
                    cv2.imwrite(crop_path, crop_img)

                    print("\nCaptured: {}".format(crop_path))

                    try:
                        run_dir = run_foodsam_on_image(
                            crop_path,
                            args.output_dir,
                            args.python_exec,
                            args.semantic_script,
                        )
                        mask_path = choose_mask_path(run_dir)
                        stats = compute_waste(mask_path, id_to_name, allowed_ids, roi_circle=crop_circle)
                        log_debug(
                            args.debug,
                            (
                                "Denominator check: frame_pixels={}, roi_pixels={} ({:.2f}% of frame), "
                                "total_food_pct={:.2f}%, unexpected_pct={:.2f}%"
                            ).format(
                                stats["frame_pixels"],
                                stats["roi_pixels"],
                                stats["roi_pct_of_frame"],
                                stats["total_food_pct"],
                                stats["unexpected_pct"],
                            ),
                        )

                        payload = {
                            "captured_at": capture_ts.isoformat(timespec="seconds"),
                            "station_id": args.station_id,
                            "menu_date": menu_date,
                            "menu_items": menu_items,
                            "image_path": image_path,
                            "crop_path": crop_path,
                            "run_dir": run_dir,
                            "mask_path": mask_path,
                            "roi_pixels": stats["roi_pixels"],
                            "total_food_pct": stats["total_food_pct"],
                            "unexpected_pct": stats["unexpected_pct"],
                            "rows": stats["rows"],
                        }
                        capture_id = save_capture(conn, payload)
                        print_summary(capture_id, stats)
                    except Exception as exc:
                        print("Capture processing error: {}".format(exc))

                    ready_for_capture = False
                    stable_window.clear()
                    last_capture_ts = time.time()
            else:
                stable_window.clear()

            if args.show_preview:
                preview = draw_preview(frame, circle, ready_for_capture, menu_text, args.arm_key)
                cv2.imshow("Canteen Auto Capture", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(args.arm_key.lower()):
                    ready_for_capture = True
                    stable_window.clear()
                    print("Capture armed")
                if key == ord("q"):
                    break

    finally:
        cap.release()
        conn.close()
        if args.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
