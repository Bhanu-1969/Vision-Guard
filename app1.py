import cv2
import numpy as np
from ultralytics import YOLO

# --- Constants ---
VIDEO_SOURCE = "video1.avi"  # Path to your video file

# --- Global Variables ---
# A list to store the points of the polygon drawn by the user
polygon_points = []

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function to capture clicks for drawing the polygon.
    """
    global polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"Point added: ({x}, {y}). Press 'Enter' to finalize.")

def main():
    """
    Main function to run the zone definition and object detection.
    """
    global polygon_points
    
    # ====================================================================
    # Phase 1: Let the user define the restricted zone
    # ====================================================================
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_SOURCE}")
        return

    # Read the first frame to use for drawing the zone
    success, frame = cap.read()
    if not success:
        print("Error: Could not read the first frame from the video.")
        cap.release()
        return

    window_name_draw = 'Define Restricted Zone - Press ENTER to Confirm'
    cv2.namedWindow(window_name_draw)
    cv2.setMouseCallback(window_name_draw, mouse_callback)
    
    print("Please define the restricted zone by clicking on the image.")
    print("Press 'c' to clear points. Press 'Enter' to confirm the zone.")

    while True:
        temp_frame = frame.copy()
        
        # Draw the polygon-in-progress
        if len(polygon_points) > 1:
            cv2.polylines(temp_frame, [np.array(polygon_points)], isClosed=False, color=(0, 255, 0), thickness=2)
        
        # Draw the points
        for point in polygon_points:
            cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
            
        cv2.imshow(window_name_draw, temp_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # 13 is the Enter key
            break
        if key == ord('c'): # Press 'c' to clear points
            print("Cleared all points.")
            polygon_points = []

    cv2.destroyAllWindows()
    
    if len(polygon_points) < 3:
        print("\nError: A polygon must have at least 3 points. Exiting.")
        cap.release()
        return

    restricted_area = np.array(polygon_points, np.int32)
    print(f"\nZone defined successfully with {len(polygon_points)} points.")
    print("Starting object detection...")

    # ====================================================================
    # Phase 2: Run YOLO detection with the defined zone
    # ====================================================================

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Reset video to the beginning for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)
        
        # Draw the restricted zone on the frame
        cv2.polylines(frame, [restricted_area], isClosed=True, color=(255, 0, 0), thickness=2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                
                # Calculate the bottom-center point of the bounding box
                bottom_center_x = int((x1 + x2) / 2)
                bottom_center_y = int(y2)
                
                # Check if the object's point is inside the polygon
                is_inside = cv2.pointPolygonTest(restricted_area, (bottom_center_x, bottom_center_y), False)
                
                color = (0, 255, 0)  # Green for objects outside
                
                if is_inside > 0:
                    color = (0, 0, 255)  # Red for objects inside
                    # --- YOUR ALERT/NOTIFICATION LOGIC CAN GO HERE ---
                
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID:{track_id}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw the reference point on the object
                cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, color, -1)
                
        cv2.imshow("Restricted Area Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()