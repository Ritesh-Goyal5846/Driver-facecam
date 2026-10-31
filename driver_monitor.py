import os
import wave
import urllib.request
import threading
import winsound
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO


class AlertSoundManager:
    """
    Synthesizes high-decibel, multi-harmonic 16-bit WAV audio directly on startup:
    1) phone_reminder.wav: Soft, clean dual-chime reminder (played at 25% amplitude).
    2) critical_jolt.wav: Maximum-amplitude dissonant square-wave siren + rapid staccato
       percussive blasts designed to physically jolt a sleeping driver awake.
    """
    def __init__(self):
        self.phone_wav = "phone_reminder.wav"
        self.critical_wav = "critical_jolt.wav"
        self._build_phone_chime()
        self._build_critical_jolt_alarm()

        self.mode = "NONE"  # "NONE", "PHONE", or "CRITICAL"
        self.muted = False
        self.running = True
        self.siren_playing = False
        self.last_phone_beep = 0.0
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()

    def _save_wav(self, filename, samples, sample_rate=44100):
        clipped = np.clip(samples, -1.0, 1.0)
        pcm16 = (clipped * 32767).astype(np.int16)
        with wave.open(filename, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

    def _build_phone_chime(self):
        """Generates a polite, soft 'ding-dong' reminder chime (low volume)."""
        sr = 44100
        t1 = np.linspace(0, 0.10, int(sr * 0.10), endpoint=False)
        t2 = np.linspace(0, 0.14, int(sr * 0.14), endpoint=False)

        # Soft decaying sine waves at 784Hz (G5) and 659Hz (E5) at 25% volume
        tone1 = 0.25 * np.sin(2 * np.pi * 784 * t1) * np.exp(-12 * t1)
        tone2 = 0.25 * np.sin(2 * np.pi * 659 * t2) * np.exp(-10 * t2)
        audio = np.concatenate([tone1, tone2])
        self._save_wav(self.phone_wav, audio, sr)

    def _build_critical_jolt_alarm(self):
        """
        Generates an aggressive, maximum-loudness wake-up jolt:
        Combines a dissonant dual-oscillator square-wave siren sweep (1000Hz -> 2900Hz),
        a sub-bass punch (190Hz) to vibrate laptop speakers, and rapid staccato gating.
        """
        sr = 44100
        duration = 0.65
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Fast rising & falling frequency sweep (1000 Hz to 2900 Hz)
        sweep_freq = 1000 + 1900 * np.abs(np.sin(2 * np.pi * 3.5 * t))
        phase1 = 2 * np.pi * np.cumsum(sweep_freq) / sr

        # Dissonant tritone second oscillator (1.414x frequency) for jarring urgency
        phase2 = 2 * np.pi * np.cumsum(sweep_freq * 1.414) / sr

        # Square waves pack maximum RMS acoustic energy (much louder than sine waves)
        osc1 = np.sign(np.sin(phase1))
        osc2 = np.sign(np.sin(phase2))
        rock_punch = np.sign(np.sin(2 * np.pi * 190 * t))  # Low-end speaker punch

        raw_wave = 0.50 * osc1 + 0.35 * osc2 + 0.15 * rock_punch

        # Rapid staccato gating (14 pulses per second) to jolt the brain awake
        staccato_gate = np.where(np.sin(2 * np.pi * 14 * t) > -0.3, 1.0, 0.15)
        jolt_audio = np.clip(raw_wave * staccato_gate * 1.3, -1.0, 1.0)
        self._save_wav(self.critical_wav, jolt_audio, sr)

    def set_mode(self, status_text):
        if self.muted:
            self.mode = "NONE"
            return
        if "CRITICAL" in status_text:
            self.mode = "CRITICAL"
        elif "HAZARD" in status_text:
            self.mode = "PHONE"
        else:
            self.mode = "NONE"

    def _audio_loop(self):
        while self.running:
            if self.mode == "CRITICAL":
                if not self.siren_playing:
                    # Play continuously in hardware loop with zero gap until eyes open
                    winsound.PlaySound(
                        self.critical_wav,
                        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
                    )
                    self.siren_playing = True
                time.sleep(0.04)

            elif self.mode == "PHONE":
                if self.siren_playing:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                    self.siren_playing = False

                now = time.time()
                if (now - self.last_phone_beep) >= 2.5:
                    winsound.PlaySound(
                        self.phone_wav,
                        winsound.SND_FILENAME | winsound.SND_ASYNC
                    )
                    self.last_phone_beep = now
                time.sleep(0.05)

            else:
                if self.siren_playing:
                    # Instantly cut the siren the millisecond the driver wakes up
                    winsound.PlaySound(None, winsound.SND_PURGE)
                    self.siren_playing = False
                time.sleep(0.05)

    def stop(self):
        self.running = False
        winsound.PlaySound(None, winsound.SND_PURGE)


class EdgeDriverMonitor:
    def __init__(self):
        # 1. Auto-download Face & Hand Landmarker models if missing
        face_model = "face_landmarker.task"
        if not os.path.exists(face_model):
            print("Downloading MediaPipe face_landmarker.task...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                face_model
            )

        hand_model = "hand_landmarker.task"
        if not os.path.exists(hand_model):
            print("Downloading MediaPipe hand_landmarker.task (7.5 MB)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                hand_model
            )

        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_model),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)

        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.50,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)
        self.start_ts = time.perf_counter()

        # 2. Lightweight YOLO11n (Nano) to keep CPU load minimal
        self.yolo = YOLO("yolo11n.pt")
        self.yolo_lock = threading.Lock()
        self.yolo_busy = False
        self.phone_detected = False
        self.phone_mode = ""
        self.phone_boxes = []
        self.hand_boxes = []
        self.phone_streak = 0

        # 3. CLAHE for Lighting Normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
        self.LEFT_EAR_IDX = 234
        self.RIGHT_EAR_IDX = 454

        # Smoothing & Baseline States (EAR, MAR, Pitch)
        self.baseline_ear = 0.30
        self.baseline_pitch = 0.0

        self.ear_smooth = 0.30
        self.mar_smooth = 0.10
        self.pitch_smooth = 0.0
        self.alpha = 0.35

        # Hazard Timers
        self.eye_closed_start = None
        self.head_nod_start = None

        self.frame_count = 0
        self.latest_face_meta = None

        # Calibration State
        self.calibrating = False
        self.calib_start_time = 0.0
        self.calib_ear_samples = []
        self.calib_pitch_samples = []

    def start_calibration(self):
        """Calibrates open-eye EAR and neutral Head Pitch."""
        self.calibrating = True
        self.calib_start_time = time.time()
        self.calib_ear_samples = []
        self.calib_pitch_samples = []

    def _is_held_by_hand(self, phone_box, hand_info):
        px1, py1, px2, py2 = phone_box
        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        pw, ph = px2 - px1, py2 - py1
        phone_diag = np.hypot(pw, ph)

        hx1, hy1, hx2, hy2 = hand_info["bbox"]
        hand_diag = hand_info["diag"]

        if not (0.35 * hand_diag <= phone_diag <= 1.85 * hand_diag):
            return False

        short_side = max(1, min(pw, ph))
        long_side = max(pw, ph)
        if (long_side / short_side) > 4.2:
            return False

        pad = 12
        return (hx1 - pad <= pcx <= hx2 + pad) and (hy1 - pad <= pcy <= hy2 + pad)

    def _run_hoi_async(self, full_frame, face_meta):
        try:
            h, w, _ = full_frame.shape
            rgb_full = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_full)

            hand_res = self.hand_landmarker.detect(mp_img)
            detected_hands = []

            if hand_res.hand_landmarks:
                for h_lm in hand_res.hand_landmarks:
                    xs = [int(pt.x * w) for pt in h_lm]
                    ys = [int(pt.y * h) for pt in h_lm]
                    hx1, hy1, hx2, hy2 = min(xs), min(ys), max(xs), max(ys)
                    hand_diag = np.hypot(hx2 - hx1, hy2 - hy1) + 1e-6

                    wrist = (int(h_lm[0].x * w), int(h_lm[0].y * h))
                    index_mcp = (int(h_lm[5].x * w), int(h_lm[5].y * h))
                    thumb_tip = (int(h_lm[4].x * w), int(h_lm[4].y * h))
                    index_tip = (int(h_lm[8].x * w), int(h_lm[8].y * h))

                    pinch_ratio = np.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1]) / hand_diag
                    is_grip = 0.22 < pinch_ratio < 0.85

                    detected_hands.append({
                        "bbox": (hx1, hy1, hx2, hy2),
                        "diag": hand_diag,
                        "wrist": wrist,
                        "index_mcp": index_mcp,
                        "is_grip": is_grip
                    })

            candidate_boxes = []
            mode_label = ""

            if detected_hands:
                res = self.yolo.predict(full_frame, imgsz=480, classes=[65, 67], conf=0.25, verbose=False)

                phone_candidates = []
                remote_boxes = []
                for box in res[0].boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    if cls_id == 65:
                        remote_boxes.append((x1, y1, x2, y2, conf))
                    elif cls_id == 67:
                        phone_candidates.append((x1, y1, x2, y2, conf))

                for (px1, py1, px2, py2, conf) in phone_candidates:
                    is_actually_remote = False
                    for (rx1, ry1, rx2, ry2, rconf) in remote_boxes:
                        if abs(px1 - rx1) < 30 and rconf >= conf * 0.85:
                            is_actually_remote = True
                            break
                    if is_actually_remote:
                        continue

                    for hand in detected_hands:
                        if self._is_held_by_hand((px1, py1, px2, py2), hand):
                            if conf >= 0.42 or (conf >= 0.25 and hand["is_grip"]):
                                candidate_boxes.append((px1, py1, px2, py2))
                                mode_label = f"HANDHELD PHONE ({int(conf*100)}%)"
                                break

                if not candidate_boxes and face_meta is not None:
                    l_ear, r_ear, face_w = face_meta["l_ear"], face_meta["r_ear"], face_meta["face_w"]
                    ear_thresh = face_w * 0.55
                    for hand in detected_hands:
                        hx, hy = hand["index_mcp"]
                        dist_l = np.hypot(hx - l_ear[0], hy - l_ear[1])
                        dist_r = np.hypot(hx - r_ear[0], hy - r_ear[1])
                        if min(dist_l, dist_r) < ear_thresh and hand["wrist"][1] > hy:
                            candidate_boxes.append(hand["bbox"])
                            mode_label = "PHONE CALL (EAR)"
                            break

            with self.yolo_lock:
                self.hand_boxes = [h["bbox"] for h in detected_hands]
                if candidate_boxes:
                    self.phone_streak = min(5, self.phone_streak + 1)
                    if self.phone_streak >= 2:
                        self.phone_detected = True
                        self.phone_boxes = candidate_boxes
                        self.phone_mode = mode_label
                else:
                    self.phone_streak = max(0, self.phone_streak - 1)
                    if self.phone_streak == 0:
                        self.phone_detected = False
                        self.phone_boxes = []
                        self.phone_mode = ""
        finally:
            self.yolo_busy = False

    def _euclidean(self, p1, p2):
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def compute_ear(self, landmarks, indices, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        v1 = self._euclidean(pts[1], pts[5])
        v2 = self._euclidean(pts[2], pts[4])
        horiz = self._euclidean(pts[0], pts[3])
        return (v1 + v2) / (2.0 * horiz + 1e-6), pts

    def compute_mar(self, landmarks, indices, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        v1 = self._euclidean(pts[1], pts[7])
        v2 = self._euclidean(pts[2], pts[6])
        v3 = self._euclidean(pts[3], pts[5])
        horiz = self._euclidean(pts[0], pts[4])
        return (v1 + v2 + v3) / (3.0 * horiz + 1e-6), pts

    def compute_head_pitch(self, lm, w, h):
        """
        Computes vertical 3D Head Pitch (nodding down) in degrees using
        Nose Tip (1), Forehead (10), and Chin (152).
        """
        nose = np.array([lm[1].x * w, lm[1].y * h, lm[1].z * w])
        forehead = np.array([lm[10].x * w, lm[10].y * h, lm[10].z * w])
        chin = np.array([lm[152].x * w, lm[152].y * h, lm[152].z * w])

        face_height = np.linalg.norm(chin[:2] - forehead[:2]) + 1e-6
        pitch_deg = np.degrees(np.arctan2(chin[2] - forehead[2], face_height)) * 2.0

        nose_2d = (int(nose[0]), int(nose[1]))
        return pitch_deg, nose_2d

    def process_frame(self, frame):
        t0 = time.perf_counter()
        h, w, _ = frame.shape

        inf_w, inf_h = 480, 360
        small = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)

        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
        rgb_small = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
        timestamp_ms = int((time.perf_counter() - self.start_ts) * 1000)
        results = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        status = "NO DRIVER DETECTED"
        rel_pitch = 0.0

        if results.face_landmarks:
            lm = results.face_landmarks[0]

            l_ear_pt = (int(lm[self.LEFT_EAR_IDX].x * w), int(lm[self.LEFT_EAR_IDX].y * h))
            r_ear_pt = (int(lm[self.RIGHT_EAR_IDX].x * w), int(lm[self.RIGHT_EAR_IDX].y * h))
            face_width = self._euclidean(l_ear_pt, r_ear_pt)
            self.latest_face_meta = {"l_ear": l_ear_pt, "r_ear": r_ear_pt, "face_w": face_width}

            # 1. Compute EAR, MAR, and Vertical Head Pitch
            left_ear, left_pts = self.compute_ear(lm, self.LEFT_EYE, w, h)
            right_ear, right_pts = self.compute_ear(lm, self.RIGHT_EYE, w, h)
            raw_ear = (left_ear + right_ear) / 2.0
            raw_mar, mouth_pts = self.compute_mar(lm, self.MOUTH, w, h)
            raw_pitch, nose_pt = self.compute_head_pitch(lm, w, h)

            self.ear_smooth = self.alpha * raw_ear + (1 - self.alpha) * self.ear_smooth
            self.mar_smooth = self.alpha * raw_mar + (1 - self.alpha) * self.mar_smooth
            self.pitch_smooth = self.alpha * raw_pitch + (1 - self.alpha) * self.pitch_smooth

            rel_pitch = self.pitch_smooth - self.baseline_pitch

            cv2.polylines(frame, [np.array(left_pts, dtype=np.int32)], True, (0, 255, 255), 1)
            cv2.polylines(frame, [np.array(right_pts, dtype=np.int32)], True, (0, 255, 255), 1)
            cv2.polylines(frame, [np.array(mouth_pts, dtype=np.int32)], True, (255, 255, 0), 1)

            end_y = int(nose_pt[1] + rel_pitch * 3.5)
            cv2.arrowedLine(frame, nose_pt, (nose_pt[0], end_y), (255, 0, 255), 2, tipLength=0.25)

            # 2. Calibration Mode
            if self.calibrating:
                elapsed = time.time() - self.calib_start_time
                if elapsed <= 5.0:
                    self.calib_ear_samples.append(raw_ear)
                    self.calib_pitch_samples.append(raw_pitch)
                    status = f"CALIBRATING... {max(0.0, 5.0 - elapsed):.1f}s"
                else:
                    if self.calib_ear_samples:
                        self.baseline_ear = float(np.mean(self.calib_ear_samples))
                        self.baseline_pitch = float(np.mean(self.calib_pitch_samples))
                    self.calibrating = False
                    status = "OPTIMAL"
            else:
                status = "OPTIMAL"
                now = time.time()
                ear_threshold = self.baseline_ear * 0.70

                # Check 1: Micro-Sleep (Eyes closed > 1.2s)
                if self.ear_smooth < ear_threshold:
                    if self.eye_closed_start is None:
                        self.eye_closed_start = now
                    elif (now - self.eye_closed_start) > 1.2:
                        status = "CRITICAL: MICRO-SLEEP"
                else:
                    self.eye_closed_start = None

                # Check 2: Head Nodding Off (Pitch drops down > 18 degrees for > 3.0 seconds)
                if rel_pitch > 18.0:
                    if self.head_nod_start is None:
                        self.head_nod_start = now
                    elif (now - self.head_nod_start) > 3.0 and "CRITICAL" not in status:
                        status = "CRITICAL: HEAD NODDING OFF"
                else:
                    self.head_nod_start = None

                # Check 3: Yawning
                if status == "OPTIMAL" and self.mar_smooth > 0.60:
                    status = "WARNING: FATIGUE (YAWN)"
        else:
            self.latest_face_meta = None

        # 3. Dispatch Hand + YOLO11n Fusion to background thread every 4 frames
        self.frame_count += 1
        if self.frame_count % 4 == 0 and not self.yolo_busy:
            self.yolo_busy = True
            threading.Thread(
                target=self._run_hoi_async,
                args=(frame.copy(), self.latest_face_meta),
                daemon=True
            ).start()

        with self.yolo_lock:
            current_phones = list(self.phone_boxes)
            current_hands = list(self.hand_boxes)
            phone_det = self.phone_detected
            phone_mode = self.phone_mode

        for (hx1, hy1, hx2, hy2) in current_hands:
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (100, 255, 100), 1)

        for (x1, y1, x2, y2) in current_phones:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, phone_mode, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        if phone_det and status == "OPTIMAL":
            status = f"HAZARD: {phone_mode}"

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "ear": round(self.ear_smooth, 3),
            "threshold": round(self.baseline_ear * 0.70, 3),
            "mar": round(self.mar_smooth, 3),
            "pitch": round(rel_pitch, 1),
            "phone": phone_det,
            "phone_mode": phone_mode,
            "status": status,
            "latency_ms": round(latency_ms, 1)
        }


def draw_hud(frame, telemetry, fps, muted):
    h, w, _ = frame.shape
    overlay = frame.copy()

    status = telemetry["status"]
    if "CRITICAL" in status:
        banner_color = (0, 0, 230)
    elif "WARNING" in status or "HAZARD" in status:
        banner_color = (0, 140, 255)
    elif "CALIBRATING" in status:
        banner_color = (255, 180, 0)
    elif status == "OPTIMAL":
        banner_color = (0, 180, 0)
    else:
        banner_color = (100, 100, 100)

    cv2.rectangle(overlay, (0, 0), (w, 48), banner_color, -1)
    cv2.rectangle(overlay, (12, 60), (365, 290), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"STATUS: {status}", (18, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    audio_label = "MUTED" if muted else (
        "MAX JOLT SIREN" if "CRITICAL" in status else ("SOFT CHIME" if "HAZARD" in status else "STANDBY")
    )

    metrics = [
        f"EAR: {telemetry['ear']:.3f} (Thresh: {telemetry['threshold']:.3f})",
        f"MAR: {telemetry['mar']:.3f} (Yawn > 0.60)",
        f"Head Pitch: {telemetry['pitch']:+.1f} deg (3s Nod > +18.0)",
        f"Phone Hazard: {telemetry['phone_mode'] if telemetry['phone'] else 'NONE'}",
        f"Audio Alarm: {audio_label}",
        f"Frame Latency: {telemetry['latency_ms']:.1f} ms | FPS: {int(fps)}"
    ]
    for i, text in enumerate(metrics):
        cv2.putText(frame, text, (24, 90 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (230, 230, 230), 1)

    cv2.putText(frame, "[C] 5s Calibrate   [M] Mute Audio   [Q] Quit", (18, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

    if "CRITICAL" in status:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    monitor = EdgeDriverMonitor()
    sound_mgr = AlertSoundManager()
    prev_time = time.perf_counter()
    fps_smooth = 30.0

    print("Edge Driver Monitor Running! Controls: [C] Calibrate | [M] Mute | [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        telemetry = monitor.process_frame(frame)
        sound_mgr.set_mode(telemetry["status"])

        now = time.perf_counter()
        inst_fps = 1.0 / (now - prev_time + 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps
        prev_time = now

        draw_hud(frame, telemetry, fps_smooth, sound_mgr.muted)
        cv2.imshow("Edge-First Driver Hazard & Micro-Sleep Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            monitor.start_calibration()
        elif key == ord('m'):
            sound_mgr.muted = not sound_mgr.muted

    sound_mgr.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()