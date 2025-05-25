#!/usr/bin/env python3
"""
Test script for face embeddings with Oak D Lite camera.
Press 'q' to quit, 's' to save current embedding.
"""

import cv2
import numpy as np
from src.camera import OakDLiteCamera
from src.face_detector import FaceDetector
from src.face_embeddings import FaceNetEmbedder

def main():
    # Initialize components
    camera = OakDLiteCamera(fps=20, resolution=(640, 480))
    face_detector = FaceDetector(min_detection_confidence=0.7)
    embedder = FaceNetEmbedder()
    
    if not camera.initialize():
        print("Failed to initialize camera")
        return
    
    print("System initialized. Press 'q' to quit, 's' to save embedding.")
    
    saved_embedding = None
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                # Detect and crop face
                face_crop = face_detector.crop_largest_face(frame)
                
                if face_crop is not None:
                    # Extract embedding
                    embedding = embedder.extract_embedding(face_crop)
                    
                    if embedding is not None:
                        # Display info
                        info_text = f"Embedding: {embedding.shape} | Norm: {np.linalg.norm(embedding):.3f}"
                        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Compare with saved embedding if exists
                        if saved_embedding is not None:
                            distance = embedder.compute_cosine_distance(embedding, saved_embedding)
                            distance_text = f"Distance from saved: {distance:.3f}"
                            cv2.putText(frame, distance_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    cv2.imshow("Cropped Face", face_crop)
                
                cv2.imshow("Camera Feed", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and face_crop is not None:
                embedding = embedder.extract_embedding(face_crop)
                if embedding is not None:
                    saved_embedding = embedding.copy()
                    print("Embedding saved!")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 