#!/usr/bin/env python3
"""
Test script for dataset loading and FAISS search.
"""

import cv2
from src.face_embeddings import FaceNetEmbedder
from src.dataset_manager import DatasetManager

def main():
    print("Testing dataset loading and FAISS search...")
    
    # Initialize embedder
    embedder = FaceNetEmbedder()
    
    # Initialize dataset manager
    dataset_path = "archive/celeba_hq"
    dataset_manager = DatasetManager(dataset_path, embedder)
    
    # Load dataset (limit to 500 for testing)
    print("Loading dataset...")
    if not dataset_manager.load_dataset(max_images=500):
        print("Failed to load dataset")
        return
    
    # Extract embeddings
    print("Extracting embeddings...")
    if not dataset_manager.extract_embeddings():
        print("Failed to extract embeddings")
        return
    
    # Build FAISS index
    print("Building FAISS index...")
    if not dataset_manager.build_faiss_index():
        print("Failed to build FAISS index")
        return
    
    # Save for future use
    print("Saving dataset index...")
    dataset_manager.save_index("dataset_index.pkl")
    
    # Test search with first image
    if dataset_manager.face_images:
        test_image = dataset_manager.face_images[0]
        test_embedding = embedder.extract_embedding(test_image)
        
        if test_embedding is not None:
            # Find farthest face
            results = dataset_manager.find_farthest_face(test_embedding, k=3)
            
            print(f"\nFound {len(results)} farthest faces:")
            for i, (idx, distance, image) in enumerate(results):
                print(f"  {i+1}. Index: {idx}, Distance: {distance:.3f}")
                
                # Display results
                cv2.imshow(f"Query Image", test_image)
                cv2.imshow(f"Farthest Face {i+1}", image)
            
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("Dataset test complete!")

if __name__ == "__main__":
    main() 