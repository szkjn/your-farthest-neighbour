#!/usr/bin/env python3
"""
Test script for Oak D Lite camera functionality.
Press 'q' to quit.
"""

import cv2
from src.camera import OakDLiteCamera

def main():
    # Initialize camera
    camera = OakDLiteCamera(fps=20, resolution=(640, 480))
    
    if not camera.initialize():
        print("Failed to initialize camera")
        return
    
    print("Camera initialized. Press 'q' to quit.")
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow("Oak D Lite - RGB", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 