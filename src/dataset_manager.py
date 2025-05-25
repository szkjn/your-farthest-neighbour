import os
import cv2
import numpy as np
import faiss
import pickle
from typing import List, Tuple, Optional
from src.face_embeddings import FaceNetEmbedder

class DatasetManager:
    def __init__(self, dataset_path: str, embedder: FaceNetEmbedder):
        self.dataset_path = dataset_path
        self.embedder = embedder
        self.face_images = []
        self.face_embeddings = []
        self.image_paths = []
        self.faiss_index = None
        
    def load_dataset(self, max_images: int = 1000) -> bool:
        """Load face images from dataset directory."""
        if not os.path.exists(self.dataset_path):
            print(f"Dataset path does not exist: {self.dataset_path}")
            return False
        
        print(f"Loading dataset from {self.dataset_path}...")
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        count = 0
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if count >= max_images:
                    break
                    
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_path = os.path.join(root, file)
                    
                    # Load and resize image
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Resize to standard face size
                        image = cv2.resize(image, (224, 224))
                        
                        self.face_images.append(image)
                        self.image_paths.append(image_path)
                        count += 1
                        
                        if count % 100 == 0:
                            print(f"Loaded {count} images...")
            
            if count >= max_images:
                break
        
        print(f"Loaded {len(self.face_images)} images from dataset")
        return len(self.face_images) > 0
    
    def extract_embeddings(self) -> bool:
        """Extract embeddings for all loaded images."""
        if not self.face_images:
            print("No images loaded. Call load_dataset() first.")
            return False
        
        print("Extracting embeddings...")
        self.face_embeddings = []
        
        for i, image in enumerate(self.face_images):
            embedding = self.embedder.extract_embedding(image)
            if embedding is not None:
                self.face_embeddings.append(embedding)
            else:
                # Remove failed image
                self.face_images.pop(i)
                self.image_paths.pop(i)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(self.face_images)} images")
        
        print(f"Extracted {len(self.face_embeddings)} embeddings")
        return len(self.face_embeddings) > 0
    
    def build_faiss_index(self) -> bool:
        """Build FAISS index for fast similarity search."""
        if not self.face_embeddings:
            print("No embeddings available. Call extract_embeddings() first.")
            return False
        
        print("Building FAISS index...")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.face_embeddings).astype('float32')
        
        # Create FAISS index (L2 distance for maximum distance search)
        dimension = embeddings_array.shape[1]  # Should be 128
        self.faiss_index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings_array)
        
        print(f"FAISS index built with {self.faiss_index.ntotal} embeddings")
        return True
    
    def find_farthest_face(self, query_embedding: np.ndarray, k: int = 1) -> List[Tuple[int, float, np.ndarray]]:
        """
        Find the farthest face(s) from query embedding.
        Returns: List of (index, distance, image) tuples
        """
        if self.faiss_index is None:
            print("FAISS index not built. Call build_faiss_index() first.")
            return []
        
        # Search for k nearest neighbors (we'll take the farthest)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search all faces to find maximum distance
        distances, indices = self.faiss_index.search(query_embedding, self.faiss_index.ntotal)
        
        # Get the farthest k faces (highest distances)
        farthest_indices = np.argsort(distances[0])[-k:][::-1]
        
        results = []
        for idx in farthest_indices:
            original_idx = indices[0][idx]
            distance = distances[0][idx]
            image = self.face_images[original_idx]
            results.append((original_idx, distance, image))
        
        return results
    
    def save_index(self, save_path: str) -> bool:
        """Save dataset and FAISS index to disk."""
        try:
            data = {
                'face_images': self.face_images,
                'face_embeddings': self.face_embeddings,
                'image_paths': self.image_paths
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save FAISS index separately
            if self.faiss_index:
                faiss.write_index(self.faiss_index, save_path.replace('.pkl', '.faiss'))
            
            print(f"Dataset saved to {save_path}")
            return True
            
        except Exception as e:
            print(f"Failed to save dataset: {e}")
            return False
    
    def load_index(self, load_path: str) -> bool:
        """Load dataset and FAISS index from disk."""
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            self.face_images = data['face_images']
            self.face_embeddings = data['face_embeddings']
            self.image_paths = data['image_paths']
            
            # Load FAISS index
            faiss_path = load_path.replace('.pkl', '.faiss')
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            
            print(f"Dataset loaded from {load_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return False 