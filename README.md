# Meet Your Farthest Neighbour

Real-time installation that finds the "farthest neighbor face" from a dataset using Oak D Lite camera.

## Setup

```bash
pip install -r requirements.txt
```

## Architecture

- Camera: Oak D Lite
- Face Detection: MediaPipe
- Embeddings: FaceNet (128D)
- Search: FAISS maximum cosine distance
- Dataset: CelebA-HQ (30K faces)

## Target Performance

- <100ms latency
- ~20 FPS real-time processing
