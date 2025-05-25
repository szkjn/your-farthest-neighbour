import cv2
import depthai as dai
import numpy as np
from typing import Optional, Tuple

class OakDLiteCamera:
    def __init__(self, fps: int = 30, resolution: Tuple[int, int] = (640, 480)):
        self.fps = fps
        self.resolution = resolution
        self.pipeline = None
        self.device = None
        self.q_rgb = None
        
    def initialize(self) -> bool:
        """Initialize Oak D Lite camera pipeline."""
        try:
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Define RGB camera
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(*self.resolution)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            cam_rgb.setFps(self.fps)
            
            # Create output
            rgb_out = self.pipeline.create(dai.node.XLinkOut)
            rgb_out.setStreamName("rgb")
            cam_rgb.preview.link(rgb_out.input)
            
            # Connect to device and start pipeline
            self.device = dai.Device(self.pipeline)
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get RGB frame from camera."""
        if not self.q_rgb:
            return None
            
        in_rgb = self.q_rgb.tryGet()
        if in_rgb is not None:
            # Get frame (already in RGB from Oak D)
            frame = in_rgb.getCvFrame()
            return frame
        return None
    
    def release(self):
        """Release camera resources."""
        if self.device:
            self.device.close()
            self.device = None 