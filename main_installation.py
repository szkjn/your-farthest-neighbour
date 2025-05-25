#!/usr/bin/env python3
"""
Main Interactive Art Installation - Real-Time Face Opposite Finder
Split-screen display: Live face (left) | Farthest neighbor (right)
"""

import cv2
import numpy as np
import time
from src.camera import OakDLiteCamera
from src.face_detector import FaceDetector
from src.face_embeddings import FaceNetEmbedder
from src.dataset_manager import DatasetManager

class FaceOppositeInstallation:
    def __init__(self, dataset_path: str = "archive/celeba_hq"):
        self.camera = OakDLiteCamera(fps=20, resolution=(640, 480))
        self.face_detector = FaceDetector(min_detection_confidence=0.7)
        self.embedder = FaceNetEmbedder()
        self.dataset_manager = DatasetManager(dataset_path, self.embedder)
        
        # Display settings
        self.display_width = 1280
        self.display_height = 480
        self.face_size = 400
        
        # Performance tracking
        self.last_embedding_time = 0
        self.embedding_interval = 0.1  # Update every 100ms for real-time feel
        
    def initialize(self) -> bool:
        """Initialize all components."""
        print("Initializing Face Opposite Finder Installation...")
        
        # Initialize camera
        if not self.camera.initialize():
            print("Failed to initialize camera")
            return False
        
        # Load or create dataset index
        if not self._load_or_create_dataset():
            print("Failed to load dataset")
            return False
        
        print("Installation ready!")
        return True
    
    def _load_or_create_dataset(self) -> bool:
        """Load existing dataset index or create new one."""
        # Try to load existing index
        if self.dataset_manager.load_index("dataset_index.pkl"):
            print("Loaded existing dataset index")
            return True
        
        # Create new index
        print("Creating new dataset index...")
        if not self.dataset_manager.load_dataset(max_images=500):
            return False
        
        if not self.dataset_manager.extract_embeddings():
            return False
        
        if not self.dataset_manager.build_faiss_index():
            return False
        
        # Save for future use
        self.dataset_manager.save_index("dataset_index.pkl")
        return True
    
    def create_split_screen(self, live_face: np.ndarray, farthest_face: np.ndarray, distance: float) -> np.ndarray:
        """Create split-screen display."""
        # Create black canvas
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Resize faces to display size
        if live_face is not None:
            live_resized = cv2.resize(live_face, (self.face_size, self.face_size))
            # Place on left side
            y_offset = (self.display_height - self.face_size) // 2
            x_offset = 50
            canvas[y_offset:y_offset+self.face_size, x_offset:x_offset+self.face_size] = live_resized
            
            # Add label
            cv2.putText(canvas, "YOU", (x_offset, y_offset-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if farthest_face is not None:
            far_resized = cv2.resize(farthest_face, (self.face_size, self.face_size))
            # Place on right side
            y_offset = (self.display_height - self.face_size) // 2
            x_offset = self.display_width - self.face_size - 50
            canvas[y_offset:y_offset+self.face_size, x_offset:x_offset+self.face_size] = far_resized
            
            # Add label with distance
            label = f"FARTHEST NEIGHBOR (d={distance:.2f})"
            cv2.putText(canvas, label, (x_offset, y_offset-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add center divider
        center_x = self.display_width // 2
        cv2.line(canvas, (center_x, 0), (center_x, self.display_height), (100, 100, 100), 2)
        
        return canvas
    
    def run(self):
        """Run the interactive installation."""
        if not self.initialize():
            return
        
        print("Starting interactive installation...")
        print("Press 'q' to quit, 'f' for fullscreen")
        
        current_farthest = None
        current_distance = 0.0
        
        try:
            while True:
                start_time = time.time()
                
                # Get camera frame
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                # Detect and crop face to consistent square
                live_face = self.face_detector.crop_largest_face(frame)
                
                # Ensure live face is same square format as dataset images
                if live_face is not None:
                    live_face = cv2.resize(live_face, (224, 224))  # Match dataset format
                
                # Update farthest neighbor periodically for real-time feel
                current_time = time.time()
                if (live_face is not None and 
                    current_time - self.last_embedding_time > self.embedding_interval):
                    
                    # Extract embedding
                    embedding = self.embedder.extract_embedding(live_face)
                    if embedding is not None:
                        # Find farthest face
                        results = self.dataset_manager.find_farthest_face(embedding, k=1)
                        if results:
                            _, distance, farthest_image = results[0]
                            current_farthest = farthest_image
                            current_distance = distance
                    
                    self.last_embedding_time = current_time
                
                # Create split-screen display
                display = self.create_split_screen(live_face, current_farthest, current_distance)
                
                # Show performance info
                processing_time = (time.time() - start_time) * 1000
                fps_text = f"Processing: {processing_time:.1f}ms"
                cv2.putText(display, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display
                cv2.imshow("Face Opposite Finder - Interactive Installation", display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    # Toggle fullscreen (basic implementation)
                    cv2.setWindowProperty("Face Opposite Finder - Interactive Installation", 
                                        cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
        except KeyboardInterrupt:
            print("Installation stopped by user")
        
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

def main():
    installation = FaceOppositeInstallation()
    installation.run()

if __name__ == "__main__":
    main() 