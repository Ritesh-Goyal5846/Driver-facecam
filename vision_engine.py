import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

class EdgeDriverMonitor:
    def __init__(self):
        # 1. MediaPipe Face Mesh (478 3D Landmarks + Iris refinement)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        # 2. Lightweight YOLO11n for Phone/Distraction Detection (COCO Class 67 = cell phone)
        self.yolo = YOLO("yolo11n.pt")
        
        # 3. CLAHE for Hackathon Stage Lighting Normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Landmark indices for Left/Right Eye (EAR) and Inner Lips (MAR)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
        
        # Dynamic Calibration & EMA Smoothing State
        self.baseline_ear = 0.30
        self.ear_smooth = 0.30
        self.mar_smooth = 0.10
        self.alpha = 0.35  # EMA smoothing factor
        self.eye_closed_start = None
        self.frame_count = 0
        self.phone_detected = False

    def preprocess_lighting(self, frame):
        """Normalizes harsh overhead shadows or dim stage lighting via CLAHE."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_norm = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_norm, a, b)), cv2.COLOR_LAB2BGR)

    def _euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def compute_ear(self, landmarks, indices, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
        v1 = self._euclidean(pts[1], pts[5])
        v2 = self._euclidean(pts[2], pts[4])
        horiz = self._euclidean(pts[0], pts[3])
        return (v1 + v2) / (2.0 * horiz + 1e-6)

    def compute_mar(self, landmarks, indices, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
        v1 = self._euclidean(pts[1], pts[7])
        v2 = self._euclidean(pts[2], pts[6])
        v3 = self._euclidean(pts[3], pts[5])
        horiz = self._euclidean(pts[0], pts[4])
        return (v1 + v2 + v3) / (3.0 * horiz + 1e-6)

    def process_frame(self, frame):
        t0 = time.perf_counter()
        h, w, _ = frame.shape
        enhanced = self.preprocess_lighting(frame)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb)
        status = "OPTIMAL"
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # 1. Calculate & Smooth EAR / MAR
            raw_ear = (self.compute_ear(lm, self.LEFT_EYE, w, h) +
                       self.compute_ear(lm, self.RIGHT_EYE, w, h)) / 2.0
            raw_mar = self.compute_mar(lm, self.MOUTH, w, h)
            
            self.ear_smooth = self.alpha * raw_ear + (1 - self.alpha) * self.ear_smooth
            self.mar_smooth = self.alpha * raw_mar + (1 - self.alpha) * self.mar_smooth
            
            # 2. Personalized Dynamic Threshold (70% of calibrated open-eye baseline)
            ear_threshold = self.baseline_ear * 0.70
            
            if self.ear_smooth < ear_threshold:
                if self.eye_closed_start is None:
                    self.eye_closed_start = time.time()
                elif (time.time() - self.eye_closed_start) > 1.2:
                    status = "CRITICAL: MICRO-SLEEP"
            else:
                self.eye_closed_start = None
                if self.mar_smooth > 0.60:
                    status = "WARNING: FATIGUE (YAWN)"

        # 3. Run YOLO11n every 4th frame on GPU for cell phone detection
        self.frame_count += 1
        if self.frame_count % 4 == 0:
            yolo_res = self.yolo.predict(frame, classes=[67], conf=0.45, verbose=False)
            self.phone_detected = len(yolo_res[0].boxes) > 0
            
        if self.phone_detected and status == "OPTIMAL":
            status = "HAZARD: PHONE DISTRACTION"

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "ear": round(self.ear_smooth, 3),
            "mar": round(self.mar_smooth, 3),
            "phone": self.phone_detected,
            "status": status,
            "latency_ms": round(latency_ms, 1)
        }