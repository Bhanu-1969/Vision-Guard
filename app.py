import cv2
from ultralytics import YOLO
import time
from math import sqrt

# --- Constants ---
VIDEO_SOURCE = "video1.avi"  # 0 for webcam, or path to a video file
STATIONARY_THRESHOLD_DIST = 2  # Max distance (pixels) an object can move to be considered stationary
TIME_THRESHOLD_SECONDS = 10  # 10 seconds
FRAMES_TO_DISAPPEAR = 30 # Number of frames to wait before removing a disappeared object
# A list of COCO class IDs you want to track as potential abandoned objects
# 24: backpack, 26: handbag, 28: suitcase, 39: bottle, 63: laptop, etc.
CLASSES_TO_TRACK = [0, 24, 26, 28, 39, 41, 63, 64, 67, 73]

# --- Initialization ---
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use a nano model for speed, can be yolov8s, yolov8m, etc.

# Open the video source
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Dictionary to store tracked objects
# Format: {object_id: {"bbox": (x,y,w,h), "centroid": (cx,cy), "timestamp": time, "disappeared_frames": 0, "alerted": False}}
tracked_objects = {}
next_object_id = 0

# --- Main Loop ---
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
# Make sure fps is not zero to avoid errors
if fps == 0:
    fps = 30 # Default to 30 if FPS cannot be read

# --- Main Loop ---
while cap.isOpened():
    frame_count += 1 # Increment frame counter
    
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, classes=CLASSES_TO_TRACK)

    current_detections = {}
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            cx, cy = int(x), int(y)
            current_detections[track_id] = ((x, y, w, h), (cx, cy))

    found_ids = []
    for track_id, (bbox, centroid) in current_detections.items():
        found_ids.append(track_id)
        
        if track_id not in tracked_objects:
            tracked_objects[track_id] = {
                "bbox": bbox,
                "centroid": centroid,
                "start_frame": frame_count, # CHANGED: Using frame_count
                "disappeared_frames": 0,
                "alerted": False
            }
        else:
            prev_centroid = tracked_objects[track_id]["centroid"]
            distance = sqrt((centroid[0] - prev_centroid[0])**2 + (centroid[1] - prev_centroid[1])**2)

            if distance > STATIONARY_THRESHOLD_DIST:
                tracked_objects[track_id]["start_frame"] = frame_count # CHANGED: Reset with frame_count
            
            tracked_objects[track_id]["bbox"] = bbox
            tracked_objects[track_id]["centroid"] = centroid
            tracked_objects[track_id]["disappeared_frames"] = 0

    ids_to_delete = []
    for obj_id in list(tracked_objects.keys()):
        if obj_id not in found_ids:
            tracked_objects[obj_id]["disappeared_frames"] += 1
            if tracked_objects[obj_id]["disappeared_frames"] > FRAMES_TO_DISAPPEAR:
                ids_to_delete.append(obj_id)
            continue
            
        # --- CHANGED: Calculate time based on frames ---
        stationary_frames = frame_count - tracked_objects[obj_id]["start_frame"]
        stationary_time = stationary_frames / fps
        
        is_alerted = tracked_objects[obj_id]["alerted"]

        if stationary_time > TIME_THRESHOLD_SECONDS and not is_alerted:
            print(f"ALERT! Object ID {obj_id} has been stationary for more than {int(TIME_THRESHOLD_SECONDS)} seconds!")
            tracked_objects[obj_id]["alerted"] = True
            
        bbox = tracked_objects[obj_id]["bbox"]
        x, y, w, h = bbox
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        color = (0, 0, 255) if is_alerted else (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID: {obj_id} Time: {int(stationary_time)}s" # This now shows correct video time
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for obj_id in ids_to_delete:
        if obj_id in tracked_objects:
            del tracked_objects[obj_id]

    cv2.imshow("Abandoned Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()