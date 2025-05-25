# Meet Your Farthest Neighbour

Real-time installation that finds the "farthest neighbor face" from a dataset using Oak D Lite camera.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CelebA-HQ Dataset

1. Go to [Kaggle CelebA-HQ Dataset](https://www.kaggle.com/datasets/lamsimon/celebahq)
2. Download the dataset (30K images, ~90GB)
3. Extract to `dataset/celeba-hq/`

### 3. Test Components

```bash
# Test camera
python test_camera.py

# Test face detection
python test_face_detection.py

# Test embeddings
python test_embeddings.py

# Test full system (after dataset download)
python test_dataset.py
```

## Architecture

### Components Built

- **Camera**: Oak D Lite RGB capture (`src/camera.py`)
- **Face Detection**: MediaPipe face detection + cropping (`src/face_detector.py`)
- **Embeddings**: FaceNet-style 128D vectors using MobileNetV2 (`src/face_embeddings.py`)
- **Dataset**: FAISS-powered maximum distance search (`src/dataset_manager.py`)

### Pipeline

1. Oak D Lite captures RGB frames (640x480 @ 20fps)
2. MediaPipe detects faces with extra forehead padding
3. Face crops resized to 224x224 maintaining aspect ratio
4. MobileNetV2 extracts 128D embeddings
5. FAISS searches for maximum cosine distance
6. Display split-screen: live face | farthest neighbor

## Target Performance

- <100ms latency
- ~20 FPS real-time processing
- Maximum cosine distance search in embedding space

## Next Steps

- Integrate all components into main application
- Implement split-screen display
- Optimize for real-time performance
- Replace MobileNetV2 with proper FaceNet weights
