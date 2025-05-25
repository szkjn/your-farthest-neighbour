import tensorflow as tf
import numpy as np
import cv2
from typing import Optional
from tensorflow.keras.models import load_model
import os

class FaceNetEmbedder:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load FaceNet model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model if provided
                self.model = load_model(self.model_path)
            else:
                # Use a simple CNN for now (we'll replace with proper FaceNet)
                self.model = self._create_simple_embedding_model()
            
            print("Face embedding model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            self.model = self._create_simple_embedding_model()
    
    def _create_simple_embedding_model(self):
        """Create a simple embedding model as placeholder."""
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        from tensorflow.keras.models import Model
        
        # Use MobileNetV2 as base (lightweight)
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Add embedding layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation=None, name='embeddings')(x)  # 128D embeddings
        
        model = Model(inputs=base_model.input, outputs=x)
        return model
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for embedding extraction."""
        # Ensure image is 224x224x3
        if face_image.shape != (224, 224, 3):
            face_image = cv2.resize(face_image, (224, 224))
        
        # Convert BGR to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face_image = face_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_image = np.expand_dims(face_image, axis=0)
        
        return face_image
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 128D embedding from face image.
        Args:
            face_image: Face image (224x224x3)
        Returns:
            128D embedding vector or None if failed
        """
        if self.model is None:
            return None
        
        try:
            # Preprocess image
            processed_face = self.preprocess_face(face_image)
            
            # Extract embedding
            embedding = self.model.predict(processed_face, verbose=0)
            
            # Normalize embedding (L2 normalization for cosine similarity)
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            return embedding[0]  # Return single embedding vector
            
        except Exception as e:
            print(f"Failed to extract embedding: {e}")
            return None
    
    def compute_cosine_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine distance between two embeddings."""
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2)
        # Convert to distance (1 - similarity for maximum distance search)
        distance = 1.0 - similarity
        return distance 