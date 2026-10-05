
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import torch
cv2.setNumThreads(1)
torch.set_num_threads(1)

import os
import sys

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # Suppress HEVC decoder warnings
os.environ["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output

import cv2
import json
import time
import threading
import gc
import subprocess
import signal
import traceback
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
IST = ZoneInfo("Asia/Kolkata")

# --- DEBUG SETTINGS ---
DEBUG_WINDOW_ENABLED = False
DEBUG_WINDOW_SCALE = 0.7

# Camera Settings
ORIGINAL_W, ORIGINAL_H = 1280, 720  # Input Resolution
PROC_W, PROC_H = 1280, 720           # Processing Resolution

# Detection Rate Configuration
TARGET_FPS = 4.0 # Set the desired processing rate per camera (e.g., 1.0 FPS)
SLEEP_DURATION = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0

# Alert Logic
EMPTY_REQUIRED_SECONDS = 3 * 60     # 3 minutes empty
PLACED_REQUIRED_SECONDS = 3 * 60    # 3 minutes occupied
MIN_DETECTIONS_FOR_OCCUPIED = 2     # Minimum jewels/items needed inside an ROI to count as placed
STORE_OPEN_TIME = "10:00"
STORE_CLOSE_TIME = "22:00"

# Gray Frame Filter
MIN_STD_DEV = 15 

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "jewel_v2.pt")
ALERTS_DIR = os.path.join(SCRIPT_DIR, "empty_racks")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "somajiguda.json")

# --- SETUP LOGGING ---
class DualLogger:
    def __init__(self, filepath, stream):
        self.stream = stream
        self.log_file = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"system_{datetime.now(IST).strftime('%Y-%m-%d')}.log")
sys.stdout = DualLogger(log_path, sys.stdout)
sys.stderr = DualLogger(log_path, sys.stderr)
# ---------------------

def signal_handler(sig, frame):
    print("\n[SHUTDOWN] Signal received. Shutting down gracefully...", flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

os.makedirs(ALERTS_DIR, exist_ok=True)
print(f"Created base alert folder: {ALERTS_DIR}", flush=True)

SITE_ID_TO_BRANCH = {}

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def is_valid_frame(frame):
    if frame is None: return False
    (mean, std_dev) = cv2.meanStdDev(frame)
    if std_dev[0] < MIN_STD_DEV and std_dev[1] < MIN_STD_DEV and std_dev[2] < MIN_STD_DEV:
        return False
    return True

def scale_polygon(points):
    return [(int(x * PROC_W / ORIGINAL_W), int(y * PROC_H / ORIGINAL_H)) for (x, y) in points]

def bbox_intersects_polygon(bbox, polygon):
    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (cx, cy), False) >= 0

def count_detections_in_polygon(detections, polygon):
    count = 0
    for box in detections:
        if bbox_intersects_polygon(box, polygon):
            count += 1
    return count

def check_working_hours():
    now = datetime.now(IST).time()
    start = datetime.strptime(STORE_OPEN_TIME, "%H:%M").time()
    end = datetime.strptime(STORE_CLOSE_TIME, "%H:%M").time()
    return start <= now <= end

def save_alert(frame, cam_id, roi_id, alert_type="empty", time_val="", polygon=None, extra_metadata=None):
    parts = cam_id.split("-")
    site_id = parts[1] if len(parts) > 1 else "0"
    branch = SITE_ID_TO_BRANCH.get(site_id, f"site_{site_id}")
    
    now = datetime.now(IST)
    event_time = time_val if time_val else now.strftime("%H:%M")
    
    folder_date = now.strftime("%d-%m-%Y") 
    file_date = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H-%M-%S")
    
    # --- VISUAL OVERLAY ---
    labels = {
        "first_placement": "First Placement",
        "placed": "Placed",
        "empty": "Empty",
        "last_removal": "Last Emptied",
        "closing_state": "Closing State"
    }
    label_text = labels.get(alert_type, alert_type.upper())
    display_text = f"{label_text} at {time_val}" if time_val else label_text

    img_h, img_w = frame.shape[:2]
    
    # Dynamically calculate font size to be consistent across different camera resolutions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, img_w / 1000.0)
    thickness = max(1, int(font_scale * 1.5))
    
    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
    
    # Dynamic padding based on resolution
    padding_x = max(10, int(img_w * 0.012))
    padding_y = max(10, int(img_h * 0.012))
    
    rect_x1 = img_w - text_w - (padding_x * 2)
    rect_y1 = padding_y
    rect_x2 = img_w - padding_x
    rect_y2 = padding_y + text_h + (padding_y * 2)

    # Use a copy of the frame to avoid modifying the original
    alert_frame = frame.copy()

    # Draw Deep Red Box (0, 0, 180)
    cv2.rectangle(alert_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 180), -1)
    
    # Draw White Text inside the box
    text_x = rect_x1 + padding_x
    text_y = rect_y1 + text_h + padding_y
    cv2.putText(alert_frame, display_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Highlight the specific zone (polygon) if provided
    if polygon is not None:
        if len(polygon) > 0 and isinstance(polygon[0][0], (int, float)):
            polys = [polygon]
        else:
            polys = polygon

        for poly in polys:
            scaled_poly = scale_polygon(poly)
            pts = np.array(scaled_poly, np.int32).reshape((-1, 1, 2))
            
            # Determine highlight color based on alert type
            if alert_type in ["empty", "closing_state"]:
                color = (0, 0, 255)  # Red for empty or items left at closing
            elif alert_type == "last_removal":
                color = (255, 0, 0)  # Blue for final removal
            else:
                color = (0, 255, 0)  # Green for placed
                
            cv2.polylines(alert_frame, [pts], True, color, 3)
            # Also add a semi-transparent overlay
            overlay = alert_frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, alert_frame, 0.8, 0, alert_frame)
    
    save_path = os.path.join(ALERTS_DIR, branch, cam_id, folder_date)
    os.makedirs(save_path, exist_ok=True)
    
    filename = f"alert-{alert_type.replace('_', '-')}_{site_id}_{cam_id}_{file_date}_{time_str}.png"
    full_image_path = os.path.join(save_path, filename)
    cv2.imwrite(full_image_path, alert_frame)
    
    # Save JSON metadata
    json_filename = filename.replace('.png', '.json')
    json_path = os.path.join(save_path, json_filename)
    
    metadata = {
        "timestamp": now.strftime('%d-%m-%Y %H:%M:%S'),
        "camera_id": cam_id,
        "site_id": site_id,
        "branch_name": branch,
        "event_type": alert_type,
        "roi_id": roi_id,
        "event_time": event_time,
        "image_file": filename
    }
    
    if extra_metadata:
        metadata.update(extra_metadata)
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[ALERT SAVED] {filename} and {json_filename}")

def draw_debug_overlay(frame, cam_id, rois, detections):
    if frame is None:
        view_frame = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)
        cv2.putText(view_frame, "Waiting for video frame...", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    else:
        view_frame = frame.copy()
    
    # 1. Draw ROIs (Blue)
    for roi_id, poly in rois.items():
        scaled_poly = scale_polygon(poly)
        pts = np.array(scaled_poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(view_frame, [pts], True, (255, 0, 0), 2)
        cv2.putText(view_frame, roi_id, (scaled_poly[0][0], scaled_poly[0][1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 2. Draw Detections (Jewelry)
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(view_frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.rectangle(view_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.putText(view_frame, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return view_frame

def show_debug_window(frame, cam_id, rois, detections):
    """Update a debug window with the latest frame. Does NOT call waitKey."""
    if not DEBUG_WINDOW_ENABLED:
        return

    view_frame = draw_debug_overlay(frame, cam_id, rois, detections)
    if DEBUG_WINDOW_SCALE != 1.0:
        view_frame = cv2.resize(view_frame, None, fx=DEBUG_WINDOW_SCALE, fy=DEBUG_WINDOW_SCALE)

    cv2.imshow(f"Debug - {cam_id}", view_frame)

def open_debug_windows(cameras):
    if not DEBUG_WINDOW_ENABLED:
        return

    for index, (cam_id, cam_data) in enumerate(cameras.items()):
        window_name = f"Debug - {cam_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Calculate spacing based on scaled resolution
        win_w = int(PROC_W * DEBUG_WINDOW_SCALE)
        win_h = int(PROC_H * DEBUG_WINDOW_SCALE)
        cv2.resizeWindow(window_name, win_w, win_h)
        
        # Grid layout: 3 columns, spacing them out so they don't overlap as much
        col = index % 3
        row = index // 3
        cv2.moveWindow(window_name, 50 + col * (win_w + 10), 50 + row * (win_h + 30))
        
        # Force window to front (common issue on macOS)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except:
            pass
            
        show_debug_window(None, cam_id, cam_data["rois"], [])
        cv2.waitKey(100) # Give extra time for window to initialize
        print(f"[DEBUG WINDOW] Opened {window_name}")

# ==========================================
# 3. PURE OPENCV STREAM LOADER
# ==========================================
class StreamLoader:
    def __init__(self, rtsp_url):
        self.rtsp_url = str(rtsp_url)
        self.is_video_file = os.path.isfile(self.rtsp_url) or self.rtsp_url.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm'))
        self.w, self.h = PROC_W, PROC_H
        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.fps = 25.0
        
        # Initialize OpenCV Capture with TCP preference for RTSP
        if not self.is_video_file:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = None

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            # Connect / Open
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.rtsp_url)
                if self.cap.isOpened():
                    if not self.is_video_file:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    else:
                        v_fps = self.cap.get(cv2.CAP_PROP_FPS)
                        if v_fps and v_fps > 0:
                            self.fps = v_fps
            
            if not self.cap.isOpened():
                time.sleep(5)
                continue

            # Read Loop
            ret, frame = self.cap.read()
            
            if ret:
                # Filter gray/corrupted frames at the source
                (_, std_dev) = cv2.meanStdDev(frame)
                if std_dev[0] < MIN_STD_DEV and std_dev[1] < MIN_STD_DEV and std_dev[2] < MIN_STD_DEV:
                    # Gray frame — skip it, don't store
                    continue
                
                resized = cv2.resize(frame, (self.w, self.h))
                with self.lock:
                    self.latest_frame = resized
                
                # If playing a local video file, pace frames to match video FPS
                if self.is_video_file:
                    time.sleep(1.0 / self.fps)
                continue

            # End of Video or Connection Loss Handler
            if self.is_video_file:
                self.running = False
                break
            else:
                self.cap.release()
                self.cap = None
                time.sleep(2)
                continue

        if self.cap:
            self.cap.release()

    def read(self):
        with self.lock:
            return self.latest_frame if self.latest_frame is not None else None
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)

# ==========================================
# 4. DETECTOR & MAIN
# ==========================================
class Detector:
    def __init__(self):
        self.lock = threading.Lock()
        
        print("\n" + "="*60, flush=True)
        print("[DETECTOR] Initializing Heavy AI Libraries... ", flush=True)
        print("[DETECTOR] -> This can take 2-4 MINUTES on your Mac.", flush=True)
        print("[DETECTOR] -> DO NOT PRESS CTRL+C! Just wait...", flush=True)
        print("="*60 + "\n", flush=True)
        
        import torch
        from ultralytics import YOLO
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"[DETECTOR] Loading YOLO on {self.device}...", flush=True)
        self.model = YOLO(MODEL_PATH)
        self.model.to(self.device)
        print("[DETECTOR] YOLO Model loaded successfully!\n", flush=True)

    def predict_batch(self, frames, imgsz=640):
        if not frames: return []
        with self.lock:
            # Run at lower resolution on the CROPPED frames for huge GPU savings
            # Use FP16 (half=True) on CUDA for 2x speedup and 50% less VRAM
            results = self.model(frames, conf=0.25, imgsz=imgsz, max_det=200, verbose=False, device=self.device, half=(self.device == 'cuda'))
        
        batch_boxes = []
        for i, res in enumerate(results):
            if res.boxes is None or len(res.boxes) == 0:
                batch_boxes.append([])
            else:
                boxes = res.boxes.xyxy.cpu().numpy().astype(int).tolist()
                batch_boxes.append(boxes)
        return batch_boxes

def main(selected_camera=None, display=False, ignore_hours=False, config_file="somajiguda.json", video_file=None):
    global DEBUG_WINDOW_ENABLED, STORE_CLOSE_TIME
    if display:
        DEBUG_WINDOW_ENABLED = True

    if video_file:
        v_path = video_file if os.path.isabs(video_file) else os.path.join(SCRIPT_DIR, video_file)
        if not os.path.exists(v_path):
            print(f"[ERROR] Video file not found: {v_path}", flush=True)
            return
        video_file = v_path
        ignore_hours = True  # Testing video files automatically ignores store operating hours

    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.", flush=True)
        return
        
    cfg_path = config_file if os.path.isabs(config_file) else os.path.join(SCRIPT_DIR, config_file)
    if not os.path.exists(cfg_path):
        print(f"Error: {cfg_path} not found.", flush=True)
        return

    with open(cfg_path) as f:
        config = json.load(f)

    global SITE_ID_TO_BRANCH
    SITE_ID_TO_BRANCH = config.pop("site_to_branch_mapping", {})

    # Filter to the requested camera(s) if specified.
    # Accepts a single id or a comma-separated list, e.g.
    #   --camera FF-1-CAM-22
    #   --camera FF-1-CAM-22,FF-2-CAM-8,SF-3-CAM-12
    if selected_camera:
        requested = [c.strip() for c in selected_camera.split(",") if c.strip()]
        missing = [c for c in requested if c not in config]
        if missing:
            print(f"[ERROR] Camera(s) not found in {os.path.basename(CONFIG_FILE)}: {missing}. "
                  f"Available: {list(config.keys())}")
            return
        config = {c: config[c] for c in requested}

    # Initialize cameras, streams, and state tracking
    cameras = {}
    for cam_id, cam_info in config.items():
        source = video_file if video_file else cam_info["rtsp"]
        if video_file:
            print(f"[VIDEO] Playing video file for {cam_id}: {video_file}", flush=True)
        else:
            print(f"[NETWORK] Connecting to RTSP Stream for {cam_id}... (If it hangs here, the camera is OFFLINE)", flush=True)
        stream = StreamLoader(source)
        
        state = {
            roi_id: {
                "occupied": False,
                "empty_since": None,
                "occupied_since": None,
                "empty_alert_sent": False,
                "last_empty_time_str": "",
                "last_empty_frame": None,
                "last_empty_alert_time_str": "",
            } for roi_id in cam_info["rois"]
        }
        
        cameras[cam_id] = {
            "stream": stream,
            "rois": cam_info["rois"],
            "state": state,
            "last_detections": [],
            "camera_state": {
                "occupied": False,
                "first_placement_done": False,
                "last_removal_alert_sent": False,
                "last_empty_time_str": "",
                "last_empty_frame": None
            }
        }

    print(f">>> System Active. Monitoring {len(cameras)} cameras with BATCH processing.", flush=True)

    # 1. Load YOLO model first (prevents Windows GUI from hanging in 'Not Responding' state)
    print(">>> Waiting for YOLO model to load (~30-60 seconds)...", flush=True)
    try:
        detector = Detector()
        print("[DETECTOR] Ready. Alerts will start appearing now.", flush=True)
    except Exception as e:
        print(f"[DETECTOR ERROR] Failed to load model: {e}", flush=True)
        return

    # 2. Start camera streams
    for cam_data in cameras.values():
        cam_data["stream"].start()

    # 3. Open debug windows after detector is ready so windows immediately receive events
    if DEBUG_WINDOW_ENABLED:
        open_debug_windows(cameras)

    last_daily_reset_date = None
    
    try:
        while True:
            if SLEEP_DURATION > 0:
                time.sleep(SLEEP_DURATION)

            now = datetime.now(IST)
            now_ts = time.time()
            is_open = True if ignore_hours else check_working_hours()
            
            # Reset daily state once before opening, not on every closed-loop cycle.
            start_time = datetime.strptime(STORE_OPEN_TIME, "%H:%M").time()
            today = now.date()
            should_reset_today = now.time() < start_time and last_daily_reset_date != today
            if should_reset_today:
                for cam_data in cameras.values():
                    cam_state = cam_data["camera_state"]
                    cam_state["occupied"] = False
                    cam_state["first_placement_done"] = False
                    cam_state["last_removal_alert_sent"] = False
                    cam_state["last_empty_time_str"] = ""
                    cam_state["last_empty_frame"] = None

                    for roi_id in cam_data["rois"]:
                        s = cam_data["state"][roi_id]
                        s["occupied"] = False
                        s["empty_since"] = None
                        s["occupied_since"] = None
                        s["empty_alert_sent"] = False
                        s["last_empty_time_str"] = ""
                        s["last_empty_frame"] = None
                        s["last_empty_alert_time_str"] = ""
                last_daily_reset_date = today
                print(f"[DAILY RESET] Cleared rack state for {today}")
            
            batch_frames = []
            batch_cam_ids = []
            
            # Auto-trigger Last Removal when video finishes
            if video_file:
                all_dead = True
                for cam_data in cameras.values():
                    if cam_data["stream"].running:
                        all_dead = False
                        break
                
                if all_dead:
                    if is_open:
                        print("[INFO] Video ended! Automatically triggering Last Removal audit...")
                        ignore_hours = False
                        STORE_CLOSE_TIME = "00:00"
                        is_open = False
                        # Do NOT continue here; let it fall through and run the audit below.
                    else:
                        # On the second pass, after audit is complete, exit.
                        break

            for cam_id, cam_data in cameras.items():
                frame = cam_data["stream"].read()

                # Always update debug window (shows "Waiting..." if frame is None)
                show_debug_window(frame, cam_id, cam_data["rois"], cam_data["last_detections"])
                
                if not is_open:
                    # Store is closed. Check one Last Removal alert per camera.
                    cam_state = cam_data["camera_state"]
                    end_time = datetime.strptime(STORE_CLOSE_TIME, "%H:%M").time()
                    if not cam_state["last_removal_alert_sent"] and now.time() >= end_time:
                        # Determine which frame and time to use
                        if cam_state["last_empty_frame"] is not None:
                            # Use the actual moment of last removal
                            target_frame = cam_state["last_empty_frame"]
                            target_time = cam_state["last_empty_time_str"]
                            label = "last_removal"
                        else:
                            # If never emptied, show the state at closing
                            target_frame = frame.copy() if frame is not None else None
                            target_time = now.strftime("%H:%M")
                            label = "closing_state" # Still saves as a final report

                        if target_frame is not None and is_valid_frame(target_frame):
                            print(f"!!! ALERT !!! {cam_id} {label.upper()} at {target_time}")
                            all_polys = list(cam_data["rois"].values())
                            save_alert(target_frame, cam_id, "camera", label, target_time, polygon=all_polys)

                        cam_state["last_removal_alert_sent"] = True
                    continue
                
                if frame is not None and is_valid_frame(frame):
                    batch_frames.append(frame)
                    batch_cam_ids.append(cam_id)

            if not is_open:
                # Pump GUI event queue even when store is closed so Windows UI does not freeze
                if DEBUG_WINDOW_ENABLED:
                    for _ in range(50):
                        key = cv2.waitKey(100) & 0xFF
                        if key in (ord("q"), 27):
                            print("[INFO] Quit key pressed.")
                            return
                else:
                    time.sleep(5)
                continue

            # Pump GUI event queue — MUST happen every iteration for macOS Cocoa
            if DEBUG_WINDOW_ENABLED:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[INFO] Quit key pressed.")
                    break

            if not batch_frames:
                continue

            # Crop and Mask frames to ROIs to save massive GPU power
            cropped_batch_frames = []
            crop_offsets = []
            
            for i, cam_id in enumerate(batch_cam_ids):
                frame = batch_frames[i]
                rois = cameras[cam_id]["rois"]
                
                # Find min/max bounds of all ROIs for this camera
                min_x, min_y = PROC_W, PROC_H
                max_x, max_y = 0, 0
                for poly in rois.values():
                    scaled_poly = scale_polygon(poly)
                    for px, py in scaled_poly:
                        min_x = min(min_x, px)
                        min_y = min(min_y, py)
                        max_x = max(max_x, px)
                        max_y = max(max_y, py)
                
                # Add padding
                pad = 30
                x1 = max(0, min_x - pad)
                y1 = max(0, min_y - pad)
                x2 = min(PROC_W, max_x + pad)
                y2 = min(PROC_H, max_y + pad)
                crop_offsets.append((x1, y1))
                
                # Crop the frame
                cropped_frame = frame[y1:y2, x1:x2].copy()
                
                # Mask out anything outside the exact polygons to prevent false positives
                mask = np.zeros(cropped_frame.shape[:2], dtype=np.uint8)
                for poly in rois.values():
                    scaled_poly = scale_polygon(poly)
                    shifted_poly = [(px - x1, py - y1) for px, py in scaled_poly]
                    pts = np.array(shifted_poly, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
                    
                masked_crop = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
                cropped_batch_frames.append(masked_crop)

            # Run Batch Inference on the smaller, cropped frames (uses ~75% less GPU)
            detections_batch = detector.predict_batch(cropped_batch_frames, imgsz=640)

            # Process Results
            for i, cam_id in enumerate(batch_cam_ids):
                frame = batch_frames[i]
                raw_detections = detections_batch[i]
                
                # Shift detections back to original frame coordinates
                x1, y1 = crop_offsets[i]
                detections = []
                for box in raw_detections:
                    bx1, by1, bx2, by2 = box
                    detections.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                
                cam_data = cameras[cam_id]
                rois = cam_data["rois"]
                state = cam_data["state"]
                cam_state = cam_data["camera_state"]

                cam_data["last_detections"] = detections
                show_debug_window(frame, cam_id, rois, detections)

                roi_detection_counts = {}
                for roi_id, poly in rois.items():
                    scaled_poly = scale_polygon(poly)
                    roi_detection_counts[roi_id] = count_detections_in_polygon(detections, scaled_poly)

                if not roi_detection_counts:
                    continue
                    
                avg_detection_count = sum(roi_detection_counts.values()) / len(roi_detection_counts)
                camera_occupied = avg_detection_count >= MIN_DETECTIONS_FOR_OCCUPIED
                camera_first_placement_sent_now = False

                if camera_occupied:
                    if not cam_state["occupied"]:
                        cam_state["occupied"] = True

                    if not cam_state["first_placement_done"]:
                        cam_state["first_placement_done"] = True
                        camera_first_placement_sent_now = True
                        print(f"!!! ALERT !!! {cam_id} FIRST PLACEMENT avg detections {avg_detection_count:.1f}")
                        extra_meta = {"avg_detections": round(avg_detection_count, 1)}
                        
                        occupied_polys = []
                        for rid, poly in rois.items():
                            if roi_detection_counts[rid] >= MIN_DETECTIONS_FOR_OCCUPIED:
                                occupied_polys.append(poly)
                                
                        save_alert(frame.copy(), cam_id, "camera", "first_placement", now.strftime("%H:%M"), polygon=occupied_polys, extra_metadata=extra_meta)
                elif cam_state["occupied"]:
                    # Transition from Occupied -> Empty
                    cam_state["occupied"] = False
                    cam_state["last_empty_time_str"] = now.strftime("%H:%M")
                    # Capture the EXACT moment it went empty to show in the closing report
                    if frame is not None:
                        cam_state["last_empty_frame"] = frame.copy()

                for roi_id, poly in rois.items():
                    detection_count = roi_detection_counts[roi_id]
                    occupied = detection_count >= MIN_DETECTIONS_FOR_OCCUPIED
                    
                    s = state[roi_id]
                    
                    if occupied:
                        s["empty_since"] = None
                        s["empty_alert_sent"] = False
                        
                        if not s["occupied"]:
                            if camera_first_placement_sent_now:
                                # Silently mark as occupied because it was included in the camera-level First Placement
                                s["occupied"] = True
                            elif s.get("occupied_since") is None:
                                s["occupied_since"] = now_ts
                            else:
                                elapsed = now_ts - s["occupied_since"]
                                if elapsed > PLACED_REQUIRED_SECONDS:
                                    s["occupied"] = True
                                    print(f"!!! ALERT !!! {cam_id} - {roi_id} PLACED for {int(elapsed)}s")
                                    extra_meta = {"detection_count": detection_count, "placed_duration_seconds": int(elapsed)}
                                    save_alert(frame.copy(), cam_id, roi_id, "placed", now.strftime("%H:%M"), polygon=poly, extra_metadata=extra_meta)
                    else:
                        s["occupied_since"] = None
                        if s["empty_since"] is None:
                            s["empty_since"] = now_ts
                            if s["occupied"]:
                                s["last_empty_time_str"] = now.strftime("%H:%M")
                        else:
                            elapsed = now_ts - s["empty_since"]
                            if elapsed > EMPTY_REQUIRED_SECONDS:
                                if s["occupied"]:
                                    s["occupied"] = False
                                    if not s["empty_alert_sent"]:
                                        # Allow Empty alerts if first placement is done OR if it's 23:59 AM or later (late setup warning)
                                        current_time_str = now.strftime("%H:%M")
                                        if cam_state["first_placement_done"] or current_time_str >= "23:59":
                                            if not s["last_empty_time_str"]:
                                                s["last_empty_time_str"] = datetime.fromtimestamp(s["empty_since"], tz=IST).strftime("%H:%M")
                                                
                                            print(f"!!! ALERT !!! {cam_id} - {roi_id} EMPTY for {int(elapsed)}s")
                                            extra_meta = {"empty_duration_seconds": int(elapsed)}
                                            
                                            # Remember the latest EMPTY proof for the
                                            # 10 PM closing package.
                                            s["last_empty_frame"] = frame.copy()
                                            s["last_empty_alert_time_str"] = now.strftime("%H:%M")
                                            
                                            save_alert(frame.copy(), cam_id, roi_id, "empty", s["last_empty_time_str"], polygon=poly, extra_metadata=extra_meta)
                                            s["empty_alert_sent"] = True

            # Refresh GUI windows with new detections immediately (Windows HighGUI support)
            if DEBUG_WINDOW_ENABLED:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[INFO] Quit key pressed.")
                    break

            # Periodic Garbage Collection (roughly every 5 mins)
            if now_ts % 300 < SLEEP_DURATION: 
                gc.collect()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[FATAL ERROR]: {e}")
        traceback.print_exc()
    finally:
        print("[SHUTDOWN] Stopping streams...")
        for cam_data in cameras.values():
            cam_data["stream"].stop()
        if DEBUG_WINDOW_ENABLED:
            cv2.destroyAllWindows()
        print("[SHUTDOWN] Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Empty Racks Monitoring System")
    parser.add_argument("--camera", type=str, help="Run specific camera(s). Single id or comma-separated list, e.g. FF-1-CAM-22 or FF-1-CAM-22,FF-2-CAM-8")
    parser.add_argument("--video", type=str, help="Path to a local MP4/video file to test instead of live RTSP stream")
    parser.add_argument("--config", type=str, default="somajiguda.json", help="Path to cameras configuration JSON (e.g. cameras.json or cameras_andhra.json)")
    parser.add_argument("--display", "--show", action="store_true", help="Display live camera window with detection overlay")
    parser.add_argument("--ignore-hours", action="store_true", help="Bypass store opening hours check for testing")
    args = parser.parse_args()
    
    main(selected_camera=args.camera, display=args.display, ignore_hours=args.ignore_hours, config_file=args.config, video_file=args.video)
