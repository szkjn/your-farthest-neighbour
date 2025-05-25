import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.7):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (< 2m), 1 for full-range
            min_detection_confidence=min_detection_confidence
        )
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image and return bounding boxes.
        Returns: List of (x, y, width, height) tuples
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                faces.append((x, y, width, height))
        
        return faces
    
    def crop_largest_face(self, image: np.ndarray, padding: float = 0.3) -> Optional[np.ndarray]:
        """
        Detect and crop the largest face from image with padding.
        Returns: Cropped face image or None if no face detected
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Find largest face by area
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Add padding (extra padding on top for forehead)
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        pad_h_top = int(h * padding * 3)  # More padding on top
        
        # Calculate padded coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h_top)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        # Resize maintaining aspect ratio
        if face_crop.size > 0:
            h_crop, w_crop = face_crop.shape[:2]
            
            # Make it square by padding the shorter dimension
            if h_crop > w_crop:
                # Pad width
                pad_left = (h_crop - w_crop) // 2
                pad_right = h_crop - w_crop - pad_left
                face_crop = cv2.copyMakeBorder(face_crop, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
            elif w_crop > h_crop:
                # Pad height
                pad_top = (w_crop - h_crop) // 2
                pad_bottom = w_crop - h_crop - pad_top
                face_crop = cv2.copyMakeBorder(face_crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            
            # Now resize square image to 224x224
            face_crop = cv2.resize(face_crop, (224, 224))
            return face_crop
        
        return None
    
    def draw_detections(self, image: np.ndarray) -> np.ndarray:
        """Draw face detection boxes on image for visualization."""
        faces = self.detect_faces(image)
        result_image = image.copy()
        
        for x, y, w, h in faces:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        return result_image 