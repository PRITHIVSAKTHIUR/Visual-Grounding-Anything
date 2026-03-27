# **[Visual-Grounding-Anything](https://huggingface.co/buckets/prithivMLmods/visual-grounding-anything)**

Visual-Grounding-Anything is a comprehensive suite of applications designed for precise object detection, pointing, and tracking in both images and videos. Leveraging the Polaris-VGA-4B model, the system provides high-accuracy spatial reasoning and temporal association across various visual tasks. The project is divided into four specialized modules: Image Detection, Image Pointing, Video Object Tracking, and Video Points Tracking. Each module is optimized for performance and reliability using advanced heuristic matching and association algorithms.

### Installation

Before running the applications, ensure you have the required dependencies installed:

```bash
pip install torch torchvision
pip install gradio==6.9.0
pip install transformers==5.3.0
pip install supervision==0.27.0.post2
pip install opencv-python==4.13.0.92
```

### Core Components

#### 1. Image Detection (image_detection_app.py)
This application identifies and localizes multiple objects within an image based on a text prompt. It outputs precise bounding boxes and masks, providing visual confirmation of detected elements.
- **Model:** Polaris-VGA-4B-Post1.0e
- **Features:** JSON-formatted detection output, interactive Gradio UI, and bright yellow annotations for high visibility.

#### 2. Image Pointer (image_pointer_app.py)
The Image Pointer specializes in pinpointing exact coordinates of objects. It is designed for tasks requiring singular pixel-level location rather than area-based detection.
- **Workflow:** Attempts to retrieve 2D points directly; falls back to bounding box centers if necessary.
- **Visuals:** Uses keypoint markers and labels to indicate the exact visual center of target objects.

#### 3. Video Object Tracking (video_object_tracking_app.py)
A robust system for maintaining object identity across video frames. It combines deep learning-based detection with a multi-stage heuristic matching engine to handle movement and occlusions.

**Multi-Stage Heuristic Matching**
- **Pass 1: Greedy IoU Matching (Primary Tracker):** This is the main association method for standard frame-to-frame object movement. It compares bounding boxes from the previous frame to current detections using the Intersection over Union (IoU) metric. Matches are sorted from highest to lowest IoU and assigned greedily. As long as the IoU exceeds a threshold, the track ID is maintained.
- **Pass 2: Euclidean Distance Fallback (Occlusion and Jitter Handler):** This fallback handles cases where Pass 1 fails due to occlusion or bounding box jitter. It considers only unmatched boxes and compares the Euclidean distance between their centers. If a new box is within a strict radius of a previous unmatched box, the system continues the track. This ensures tracking remains robust even when objects are temporarily obscured or boxes fluctuate.

#### 4. Video Points Tracking (video_points_tracking_app.py)
This module tracks specific pixel coordinates across a video sequence, creating motion trails and providing temporal continuity for point-based prompts.

**Dynamic Proximity-Based Point Association**
- **Pass 1: Dual-Tier Point Extraction (Detector):** Uses a cascaded prompt system to get precise pixel coordinates. First, a bounding box is extracted and its geometric center calculated. If the VLM fails, a fallback prompt directly requests raw [x, y] coordinates. This minimizes detection failures and improves accuracy.
- **Pass 2: Resolution-Invariant Distance Matching (Primary Tracker):** Links points frame-to-frame using Euclidean distance. A dynamic threshold based on the video frame's diagonal ensures accurate tracking across resolutions, from 480p to 4K. Closest points are paired efficiently without fixed pixel limits.
- **Pass 3: Temporal Track Patience (Flicker Handler):** Handles brief VLM misses or occlusions. If a point is not detected, its last known position is projected forward for up to n frames before terminating the track. This smooths jitter and maintains continuous motion trails.

### Model Details

All applications utilize the **Polaris-VGA-4B-Post1.0e** model, a specialized vision-language model (VLM) fine-tuned for visual grounding tasks. It supports complex text prompts and returns scaled coordinates (0-1000) for both points and bounding boxes.

### Usage

Each script can be run independently to launch a local Gradio interface:

```bash
python image_detection_app.py
python image_pointer_app.py
python video_object_tracking_app.py
python video_points_tracking_app.py
```

### Technical Notes

- **Device Support:** Automatically detects and utilizes CUDA for GPU acceleration (using bfloat16 or float16) or falls back to CPU.
- **Input Processing:** Videos are processed frame-by-frame with configurable FPS and duration limits to ensure optimal resource management.
- **Output:** The system generates annotated images and MP4 videos with overlaid masks, boxes, IDs, and motion trails.
