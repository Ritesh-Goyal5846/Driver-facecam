# Edge-First Driver Hazard & Micro-Sleep Monitor

An ultra-low-latency, 100% offline Driver Monitoring System (DMS) that detects micro-sleeps, yawning, 3D head-pose distraction, and handheld/sideways phone usage in real time without streaming video to the cloud.

## Key Engineering Challenges & How We Solved Them

### 1. MediaPipe Legacy API Deprecation on Python 3.14
* **Issue:** Standard `mp.solutions.face_mesh` pipelines throw an `AttributeError` on newer MediaPipe/Python 3.14 builds.
* **Solution:** Migrated the core vision engine to the modern `mp.tasks.vision.FaceLandmarker` (`RunningMode.VIDEO`) and `HandLandmarker` (`RunningMode.IMAGE`) C++ task graphs with automated model provisioning.

### 2. Webcam Frame Rate Bottleneck (10 FPS -> 30+ FPS)
* **Issue:** Running full-resolution `1280x720` frames through OpenCV's default Windows Media Foundation (MSMF) driver and synchronous YOLO inference caused severe frame drops (10-12 FPS).
* **Solution:** 
  * Switched camera capture to hardware-accelerated **DirectShow (`cv2.CAP_DSHOW`) + MJPG codec**.
  * Decoupled inference resolution (`480x360` with `YCrCb` CLAHE lighting normalization) from display resolution.
  * Offloaded heavy object detection to a non-blocking **asynchronous daemon thread** (`threading.Thread`), locking the main UI loop at 30+ FPS.

### 3. False-Positive Black Rectangles & Sideways Phone Blindness
* **Issue:** Standard `yolo11n` mistook wallets, remotes, and black rectangular objects for phones, while failing to detect phones held sideways (thin edge profile) or pressed flat against the ear.
* **Solution:** Built a custom **Hand-Object Interaction (HOI) Fusion Engine**:
  * **Spatial Hand Anchoring:** Rejects any YOLO phone candidate unless its center point sits physically inside a detected 21-point MediaPipe hand polygon with a valid C-grip pinch ratio.
  * **Cross-Class Suppression:** Actively tracks COCO class `65` (`remote`) alongside class `67` (`cell phone`) using `yolo11s` to suppress calculators and remotes.
  * **Ear-Call Geometry & 3-Frame Persistence:** Tracks hand knuckle proximity to the driver's ear landmarks (`234`/`454`) to catch sideways phone calls even when the phone is occluded, backed by a 3-frame temporal streak filter.

### 4. Camera-Tilt Invariance & Personal Eye Calibration
* **Issue:** Hardcoded Eye Aspect Ratio (`EAR`) and head-angle thresholds fail across different eye shapes and camera mounting heights.
* **Solution:** Added a 5-second live auto-calibration routine (`[C]` key) that samples the driver's personal baseline `EAR` and `solvePnP` 3D head orientation (`Pitch` and `Yaw` zero-point).