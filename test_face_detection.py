#!/usr/bin/env python3
"""
Test script for face detection with Oak D Lite camera.
Press 'q' to quit.
"""

import cv2
from src.camera import OakDLiteCamera
from src.face_detector import FaceDetector

def main():
    # Initialize camera and face detector
    camera = OakDLiteCamera(fps=20, resolution=(640, 480))
    face_detector = FaceDetector(min_detection_confidence=0.7)
    
    if not camera.initialize():
        print("Failed to initialize camera")
        return
    
    print("Camera and face detector initialized. Press 'q' to quit.")
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                # Draw face detections
                frame_with_faces = face_detector.draw_detections(frame)
                
                # Crop largest face
                face_crop = face_detector.crop_largest_face(frame)
                
                # Display results
                cv2.imshow("Face Detection", frame_with_faces)
                
                if face_crop is not None:
                    cv2.imshow("Cropped Face", face_crop)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 