from __future__ import annotations
import cv2
import time
import math
import threading
import numpy as np
import sys
import os

# ------------------- Enhanced Config -------------------
EYE_AR_THRESH = 0.25            # Eye Aspect Ratio threshold for "closed" (higher = more sensitive)
EYE_CLOSED_SECONDS = 2.0        # How long eyes must stay closed to alarm
HEAD_DOWN_SECONDS = 3.0         # How long head must be down to alarm - realistic sleep detection
SHOW_LANDMARKS = True           # Toggle eye landmark visualization
CAM_INDEX = 0                   # Webcam index (try 0, 1, 2 if camera not found)

# Camera quality settings
CAMERA_WIDTH = 1920             # HD resolution width
CAMERA_HEIGHT = 1080            # HD resolution height
CAMERA_FPS = 30                 # Target FPS
DISPLAY_WIDTH = 1400            # Display window width
BRIGHTNESS = 0.5               # Camera brightness (0.0-1.0)
CONTRAST = 0.5                 # Camera contrast (0.0-1.0)
SATURATION = 0.5               # Camera saturation (0.0-1.0)

# ------------------- Enhanced Alarm System -------------------
class EnhancedAlarm:
    def __init__(self):
        self._playing = False
        self._thread = None
        self._lock = threading.Lock()
        self._alarm_type = "eye_closed"
        
        # Audio system detection
        self._audio_mode = self._detect_audio_system()
        
    def _detect_audio_system(self):
        """Detect available audio system"""
        try:
            import simpleaudio as sa
            self._sa = sa
            print("[INFO] Using simpleaudio for enhanced alerts")
            return "simpleaudio"
        except ImportError:
            pass
            
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._pygame = pygame
            print("[INFO] Using pygame for audio alerts")
            return "pygame"
        except ImportError:
            pass
            
        try:
            import winsound
            self._winsound = winsound
            print("[INFO] Using winsound for alerts")
            return "winsound"
        except ImportError:
            pass
            
        print("[INFO] Using system bell for alerts")
        return "bell"

    def _create_alert_sound(self, freq=800, duration=0.3):
        """Generate alert sound wave"""
        fs = 22050
        t = np.linspace(0, duration, int(fs * duration), False)
        wave = np.sin(2 * np.pi * freq * t) * np.sin(2 * np.pi * (freq * 1.5) * t) * 0.4
        return (wave * 32767).astype(np.int16)

    def _audio_loop(self):
        """Enhanced audio loop with different patterns"""
        pattern_count = 0
        while self._playing:
            try:
                if self._audio_mode == "simpleaudio":
                    freq = 800 if pattern_count % 2 == 0 else 1200
                    audio = self._create_alert_sound(freq, 0.2)
                    stereo_audio = np.column_stack((audio, audio))
                    play_obj = self._sa.play_buffer(stereo_audio, 2, 2, 22050)
                    play_obj.wait_done()
                    time.sleep(0.1)
                    
                elif self._audio_mode == "pygame":
                    freq = 800 if pattern_count % 2 == 0 else 1200
                    audio = self._create_alert_sound(freq, 0.2)
                    sound = self._pygame.sndarray.make_sound(np.column_stack((audio, audio)))
                    sound.play()
                    time.sleep(0.3)
                    
                elif self._audio_mode == "winsound":
                    freq = 800 if pattern_count % 2 == 0 else 1200
                    try:
                        self._winsound.Beep(freq, 200)
                    except Exception:
                        print("\a", end="", flush=True)
                    time.sleep(0.1)
                    
                else:  # bell fallback
                    print("\a" * 3, end="", flush=True)
                    time.sleep(0.5)
                    
                pattern_count += 1
                
            except Exception as e:
                print(f"[WARN] Audio error: {e}")
                print("\a", end="", flush=True)
                time.sleep(0.2)

    def start(self, alarm_type="drowsiness"):
        """Start alarm with specified type"""
        with self._lock:
            if self._playing:
                return
            self._playing = True
            self._alarm_type = alarm_type
            self._thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._thread.start()
            print(f"[ALERT] {alarm_type.upper()} ALARM ACTIVATED!")

    def stop(self):
        """Stop alarm"""
        with self._lock:
            if not self._playing:
                return
            self._playing = False
            print(f"[INFO] Alarm stopped")
        
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

# ------------------- Enhanced Computer Vision Functions -------------------
LEFT_EYE_LANDMARKS = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
]
RIGHT_EYE_LANDMARKS = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
]

LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [263, 387, 385, 362, 380, 373]

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def eye_aspect_ratio(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR) from eye landmarks"""
    if len(eye_landmarks) < 6:
        return 0.0
    
    # Vertical distances
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    
    if C == 0:
        return 0.0
    
    ear = (A + B) / (2.0 * C)
    return ear

def detect_sleeping_posture(landmarks, img_width, img_height):
    """
    Highly accurate sleep posture detection that mimics real drowsiness behavior.
    Only triggers when head position genuinely resembles someone falling asleep.
    """
    try:
        # Key facial landmarks for sleep detection
        nose_tip = landmarks[1]          # Nose tip
        chin = landmarks[175]            # Chin bottom
        forehead = landmarks[10]         # Forehead center
        left_eye = landmarks[33]         # Left eye outer corner
        right_eye = landmarks[263]       # Right eye outer corner
        left_eye_inner = landmarks[133]  # Left eye inner corner
        right_eye_inner = landmarks[362] # Right eye inner corner
        left_mouth = landmarks[61]       # Left mouth corner
        right_mouth = landmarks[291]     # Right mouth corner
        
        # Convert normalized coordinates to pixels
        def to_pixel(landmark):
            return (landmark.x * img_width, landmark.y * img_height)
        
        nose_x, nose_y = to_pixel(nose_tip)
        chin_x, chin_y = to_pixel(chin)
        forehead_x, forehead_y = to_pixel(forehead)
        left_eye_x, left_eye_y = to_pixel(left_eye)
        right_eye_x, right_eye_y = to_pixel(right_eye)
        left_eye_inner_x, left_eye_inner_y = to_pixel(left_eye_inner)
        right_eye_inner_x, right_eye_inner_y = to_pixel(right_eye_inner)
        left_mouth_x, left_mouth_y = to_pixel(left_mouth)
        right_mouth_x, right_mouth_y = to_pixel(right_mouth)
        
        # Calculate face dimensions and centers
        face_height = chin_y - forehead_y
        eye_center_x = (left_eye_x + right_eye_x) / 2
        eye_center_y = (left_eye_y + right_eye_y) / 2
        eye_width = abs(right_eye_x - left_eye_x)
        mouth_center_y = (left_mouth_y + right_mouth_y) / 2
        
        # Sleep detection scoring system
        sleep_score = 0.0
        detection_reasons = []
        
        if face_height > 0 and eye_width > 0:
            
            # 1. NOSE DROP TEST - Most important indicator
            # When drowsy, nose drops significantly below eye line
            nose_drop_ratio = (nose_y - eye_center_y) / face_height
            if nose_drop_ratio > 0.20:  # Nose 20% below eyes relative to face height
                sleep_score += 2.5
                detection_reasons.append(f"Nose drop: {nose_drop_ratio:.2f}")
            
            # 2. CHIN EXTENSION TEST
            # Drowsy head causes chin to extend far below normal position
            chin_extension_ratio = (chin_y - eye_center_y) / face_height
            if chin_extension_ratio > 1.15:  # Chin extends too far
                sleep_score += 2.0
                detection_reasons.append(f"Chin extension: {chin_extension_ratio:.2f}")
            
            # 3. FACE ELONGATION TEST
            # Sleepy face appears stretched/elongated due to head angle
            face_aspect_ratio = face_height / eye_width
            if face_aspect_ratio > 1.85:  # Face too tall (head forward)
                sleep_score += 1.5
                detection_reasons.append(f"Face elongated: {face_aspect_ratio:.2f}")
            
            # 4. EYE LEVEL TILT TEST
            # When head droops, eye line becomes significantly uneven
            eye_tilt_pixels = abs(left_eye_y - right_eye_y)
            eye_tilt_ratio = eye_tilt_pixels / face_height
            if eye_tilt_ratio > 0.03:  # 3% of face height
                sleep_score += 1.0
                detection_reasons.append(f"Eye tilt: {eye_tilt_ratio:.3f}")
            
            # 5. MOUTH POSITION TEST
            # Mouth position changes relative to eyes when head drops
            mouth_eye_distance = abs(mouth_center_y - eye_center_y)
            mouth_eye_ratio = mouth_eye_distance / face_height
            if mouth_eye_ratio > 0.35 or mouth_eye_ratio < 0.12:  # Abnormal mouth position
                sleep_score += 0.8
                detection_reasons.append(f"Mouth pos: {mouth_eye_ratio:.2f}")
            
            # 6. FOREHEAD COMPRESSION TEST
            # Less forehead visible when head tilts forward
            forehead_eye_distance = eye_center_y - forehead_y
            forehead_ratio = forehead_eye_distance / face_height
            if forehead_ratio < 0.25:  # Forehead compressed
                sleep_score += 1.2
                detection_reasons.append(f"Forehead: {forehead_ratio:.2f}")
        
        # FINAL SLEEP DETECTION DECISION
        # Require high confidence to avoid false alarms
        is_sleeping = sleep_score >= 4.0  # Need at least 4.0/7.5 points
        
        # Calculate pitch angle for display
        pitch_angle = 0.0
        if face_height > 0:
            pitch_angle = ((nose_y - eye_center_y) / face_height) * 45
        
        # Log detection when sleeping is detected
        if is_sleeping and detection_reasons:
            print(f"[😴 SLEEP] Detected sleeping posture! Score: {sleep_score:.1f}/7.5")
            print(f"[😴 SLEEP] Reasons: {', '.join(detection_reasons[:3])}")
        
        return float(pitch_angle), is_sleeping, float(sleep_score)
        
    except Exception as e:
        print(f"[WARN] Sleep detection error: {e}")
        return 0.0, False, 0.0

def draw_eye_landmarks(frame, landmarks, eye_indices, color=(0, 255, 0)):
    """Draw eye landmarks on the frame"""
    try:
        for idx in eye_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * frame.shape[1])
                y = int(landmarks[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, color, -1)
                cv2.circle(frame, (x, y), 5, color, 1)
    except Exception as e:
        print(f"[WARN] Error drawing landmarks: {e}")

def enhance_frame_quality(frame):
    """Apply image enhancement for better clarity"""
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    # Merge channels back
    enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Apply slight sharpening
    kernel = np.array([[-0.1, -0.1, -0.1],
                       [-0.1,  1.8, -0.1],
                       [-0.1, -0.1, -0.1]])
    enhanced_frame = cv2.filter2D(enhanced_frame, -1, kernel)
    
    # Slight noise reduction while preserving details
    enhanced_frame = cv2.bilateralFilter(enhanced_frame, 5, 50, 50)
    
    return enhanced_frame

def initialize_camera_hd(cam_index=0):
    """Initialize camera with high-definition settings"""
    for idx in [cam_index, 0, 1, 2]:
        print(f"[INFO] Trying camera index {idx} with HD settings...")
        cap = cv2.VideoCapture(idx)
        
        if cap.isOpened():
            # Set high-definition properties
            print("[INFO] Setting camera to HD mode...")
            
            # Try different HD resolutions in order of preference
            resolutions = [
                (1920, 1080),  # Full HD
                (1280, 720),   # HD
                (1024, 768),   # XGA
                (800, 600),    # SVGA
            ]
            
            for width, height in resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Test if resolution was set successfully
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                if actual_width >= width * 0.9 and actual_height >= height * 0.9:
                    print(f"[✓] Camera resolution set to: {int(actual_width)}x{int(actual_height)}")
                    break
            
            # Set other quality parameters
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
            cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST)
            cap.set(cv2.CAP_PROP_SATURATION, SATURATION)
            
            # Enable auto-exposure and auto-focus if available
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # Auto focus
            
            # Reduce buffer size for lower latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test camera with a few frames
            for i in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    if i == 4:  # Last test frame
                        h, w = frame.shape[:2]
                        print(f"[✓] Camera {idx} working with resolution: {w}x{h}")
                        print(f"[✓] Actual FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                        return cap
                time.sleep(0.1)
            
            cap.release()
        
    print("[ERROR] Could not find any working camera")
    return None

def resize_frame_smart(frame, target_width=None):
    """Smart frame resizing with quality preservation"""
    if target_width is None:
        target_width = DISPLAY_WIDTH
    
    h, w = frame.shape[:2]
    
    if w > target_width:
        # Calculate scaling factor
        scale = target_width / w
        new_w = target_width
        new_h = int(h * scale)
        
        # Use high-quality interpolation for downscaling
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif w < target_width * 0.7:  # Upscale if too small
        scale = (target_width * 0.8) / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use high-quality interpolation for upscaling
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return frame

# ------------------- Main Application -------------------
def main():
    global SHOW_LANDMARKS
    
    print("="*70)
    print("🎯 SMART DROWSINESS DETECTION SYSTEM - HD VERSION 🎯")
    print("="*70)
    print("[INFO] Initializing high-definition system...")
    
    # Check MediaPipe availability
    try:
        import mediapipe as mp
        print("[✓] MediaPipe loaded successfully")
    except ImportError:
        print("[✗] MediaPipe not found!")
        print("Install with: pip install mediapipe")
        return False

    # Initialize MediaPipe Face Mesh with higher confidence
    mp_face_mesh = mp.solutions.face_mesh
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,  # Higher confidence for better tracking
        min_tracking_confidence=0.7
    )

    # Initialize HD camera
    print("[INFO] Setting up HD camera...")
    cap = initialize_camera_hd(CAM_INDEX)
    if cap is None:
        return False

    # Initialize alarm system
    alarm = EnhancedAlarm()
    
    # State tracking variables
    eyes_closed_start_time = None
    head_down_start_time = None
    alarm_active = False
    last_alert_type = None
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    frame_skip_counter = 0
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    print("\n" + "="*70)
    print("📋 HD SYSTEM CONFIGURATION")
    print("="*70)
    print(f"📸 Target Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"🎬 Target FPS: {CAMERA_FPS}")
    print(f"🖥️  Display Width: {DISPLAY_WIDTH}px")
    print(f"👁  Eye closure threshold: {EYE_AR_THRESH:.3f}")
    print(f"⏱  Eye closure alert time: {EYE_CLOSED_SECONDS}s")
    print(f"😴 Sleep detection: Advanced HD posture analysis")
    print(f"⏱  Sleep posture alert time: {HEAD_DOWN_SECONDS}s")
    print(f"🔍 Show landmarks: {'ON' if SHOW_LANDMARKS else 'OFF'}")
    print("="*70)
    print("🚀 HD SYSTEM READY! Press 'Q' or ESC to quit")
    print("   Controls: 'L' = toggle landmarks, 'E' = toggle enhancement")
    print("="*70)

    # Create window with better settings
    window_name = "HD Drowsiness Detection - Press Q/ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, 800)

    # Frame enhancement toggle
    enhance_enabled = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Apply quality enhancement
            if enhance_enabled:
                frame = enhance_frame_quality(frame)
            
            # Resize for display with quality preservation
            frame = resize_frame_smart(frame)
            h, w = frame.shape[:2]
            
            current_time = time.time()
            
            # Skip processing every few frames for performance if needed
            process_frame = (frame_skip_counter % 2 == 0)  # Process every 2nd frame
            frame_skip_counter += 1
            
            if process_frame:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
            else:
                results = getattr(main, '_last_results', None)
            
            # Store results for frame skipping
            main._last_results = results

            # Initialize detection variables
            ear_value = 0.0
            pitch_angle = 0.0
            head_down_detected = False
            sleep_score = 0.0
            face_detected = False

            if results and results.multi_face_landmarks:
                face_detected = True
                landmarks = results.multi_face_landmarks[0].landmark

                # Extract eye landmark coordinates
                def get_eye_points(eye_indices):
                    points = []
                    for i in eye_indices:
                        if i < len(landmarks):
                            points.append([landmarks[i].x * w, landmarks[i].y * h])
                        else:
                            points.append([0, 0])
                    return points

                left_eye_points = get_eye_points(LEFT_EYE_IDXS)
                right_eye_points = get_eye_points(RIGHT_EYE_IDXS)

                # Calculate Eye Aspect Ratios
                left_ear = eye_aspect_ratio(left_eye_points)
                right_ear = eye_aspect_ratio(right_eye_points)
                ear_value = (left_ear + right_ear) / 2.0

                # Detect sleeping posture
                pitch_angle, head_down_detected, sleep_score = detect_sleeping_posture(landmarks, w, h)

                # Draw enhanced eye landmarks if enabled
                if SHOW_LANDMARKS:
                    draw_eye_landmarks(frame, landmarks, LEFT_EYE_LANDMARKS, (0, 255, 0))
                    draw_eye_landmarks(frame, landmarks, RIGHT_EYE_LANDMARKS, (255, 0, 0))

                # 👁 EYE CLOSURE DETECTION
                if ear_value < EYE_AR_THRESH:
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = current_time
                        print(f"[👁 EYES] Closure detected (EAR: {ear_value:.3f})")
                else:
                    if eyes_closed_start_time is not None:
                        print("[👁 EYES] Eyes reopened")
                    eyes_closed_start_time = None

                # 😴 SLEEP POSTURE DETECTION
                if head_down_detected:
                    if head_down_start_time is None:
                        head_down_start_time = current_time
                else:
                    if head_down_start_time is not None:
                        print("[😴 SLEEP] Head position normalized")
                    head_down_start_time = None

                # Calculate duration timers
                eyes_closed_duration = (current_time - eyes_closed_start_time) if eyes_closed_start_time else 0.0
                head_down_duration = (current_time - head_down_start_time) if head_down_start_time else 0.0

                # 🚨 ALARM LOGIC
                should_alarm_eyes = eyes_closed_duration >= EYE_CLOSED_SECONDS
                should_alarm_head = head_down_duration >= HEAD_DOWN_SECONDS
                should_alarm = should_alarm_eyes or should_alarm_head

                # Determine alert type
                current_alert_type = None
                if should_alarm_eyes:
                    current_alert_type = "eyes_closed"
                elif should_alarm_head:
                    current_alert_type = "sleeping_posture"

                # Control alarm
                if should_alarm and not alarm_active:
                    alarm.start(current_alert_type)
                    alarm_active = True
                    last_alert_type = current_alert_type
                elif not should_alarm and alarm_active:
                    alarm.stop()
                    alarm_active = False
                    last_alert_type = None

            else:
                # No face detected - reset timers
                if eyes_closed_start_time is not None or head_down_start_time is not None:
                    print("[🔍 FACE] Lost face tracking - resetting timers")
                
                eyes_closed_start_time = None
                head_down_start_time = None
                
                if alarm_active:
                    alarm.stop()
                    alarm_active = False

            # 🎨 DRAW ENHANCED UI ELEMENTS
            
            # Semi-transparent background for metrics
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 220), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (10, 10), (500, 220), (100, 150, 200), 2)

            # Enhanced detection metrics with better fonts
            y_offset = 40
            cv2.putText(frame, f"HD EAR: {ear_value:.3f} ({'CLOSED' if ear_value < EYE_AR_THRESH else 'OPEN'})", 
                       (20, y_offset), font, 0.7, (0, 255, 255), 2)
            
            y_offset += 30
            cv2.putText(frame, f"SLEEP SCORE: {sleep_score:.1f}/7.5", 
                       (20, y_offset), font, 0.7, (0, 255, 255), 2)
            
            y_offset += 30
            resolution_text = f"RESOLUTION: {w}x{h}"
            cv2.putText(frame, resolution_text, (20, y_offset), font, 0.6, (255, 255, 0), 2)
            
            if head_down_detected:
                y_offset += 30
                cv2.putText(frame, "STATUS: SLEEPING POSTURE!", 
                           (20, y_offset), font, 0.7, (0, 0, 255), 2)

            # Enhanced timer displays
            if eyes_closed_start_time:
                duration = current_time - eyes_closed_start_time
                y_offset += 30
                color = (0, 0, 255) if duration >= EYE_CLOSED_SECONDS else (0, 255, 255)
                cv2.putText(frame, f"Eyes closed: {duration:.1f}s", 
                           (20, y_offset), font, 0.7, color, 2)

            if head_down_start_time:
                duration = current_time - head_down_start_time  
                y_offset += 30
                color = (0, 0, 255) if duration >= HEAD_DOWN_SECONDS else (0, 255, 255)
                cv2.putText(frame, f"Sleep posture: {duration:.1f}s", 
                           (20, y_offset), font, 0.7, color, 2)

            # Enhanced FPS counter
            fps_counter += 1
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time
            
            cv2.putText(frame, f"FPS: {current_fps}", (w - 120, 35), font, 0.6, (255, 255, 255), 2)
            enhancement_status = "Enhanced" if enhance_enabled else "Normal"
            cv2.putText(frame, f"Quality: {enhancement_status}", (w - 200, 65), font, 0.5, (255, 255, 255), 1)

            # Enhanced status indicator
            if face_detected:
                status_color = (0, 0, 255) if alarm_active else (0, 255, 0)
                status_text = "DROWSINESS ALERT!" if alarm_active else "HD MONITORING"
                if alarm_active and last_alert_type:
                    if last_alert_type == "eyes_closed":
                        status_text = "🚨 EYES CLOSED ALERT!"
                    elif last_alert_type == "sleeping_posture":
                        status_text = "🚨 SLEEPING DETECTED!"
            else:
                status_color = (50, 50, 255)
                status_text = "NO FACE DETECTED"
            
            cv2.putText(frame, status_text, (w - 350, 95), font, 0.7, status_color, 2)

            # 🚨 ENHANCED ALERT OVERLAY
            if alarm_active:
                # Enhanced flashing red overlay with gradient effect
                if int(current_time * 4) % 2:  # Flash at 2Hz
                    alert_overlay = frame.copy()
                    # Create gradient effect
                    for i in range(120):
                        alpha = 0.6 * (1 - i/120)
                        cv2.rectangle(alert_overlay, (0, i), (w, i+1), (0, 0, 255), -1)
                    cv2.addWeighted(frame, 0.4, alert_overlay, 0.6, 0, frame)
                
                # Enhanced alert messages with shadow effect
                alert_msg = "⚠️ DROWSINESS DETECTED ⚠️"
                if last_alert_type == "eyes_closed":
                    detail_msg = "EYES HAVE BEEN CLOSED TOO LONG!"
                elif last_alert_type == "sleeping_posture":
                    detail_msg = "SLEEPING POSTURE DETECTED - WAKE UP!"
                else:
                    detail_msg = "WAKE UP IMMEDIATELY!"
                
                # Center the messages with shadow effect
                alert_size = cv2.getTextSize(alert_msg, font, 1.0, 3)[0]
                alert_x = (w - alert_size[0]) // 2
                
                detail_size = cv2.getTextSize(detail_msg, font, 0.8, 2)[0]
                detail_x = (w - detail_size[0]) // 2
                
                # Draw shadow
                cv2.putText(frame, alert_msg, (alert_x + 2, 52), font, 1.0, (0, 0, 0), 4)
                cv2.putText(frame, detail_msg, (detail_x + 2, 92), font, 0.8, (0, 0, 0), 3)
                
                # Draw main text
                cv2.putText(frame, alert_msg, (alert_x, 50), font, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, detail_msg, (detail_x, 90), font, 0.8, (255, 255, 255), 2)

            # Enhanced instructions with better positioning
            instructions = "Q/ESC: Quit | L: Landmarks | E: Enhancement | F: Fullscreen"
            cv2.putText(frame, instructions, (20, h - 20), font, 0.5, (200, 200, 200), 1)

            # Show the enhanced frame
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 'Q' or ESC
                print("[INFO] User requested shutdown...")
                break
            elif key == ord('l') or key == ord('L'):  # Toggle landmarks
                SHOW_LANDMARKS = not SHOW_LANDMARKS
                print(f"[INFO] Landmarks: {'ON' if SHOW_LANDMARKS else 'OFF'}")
            elif key == ord('e') or key == ord('E'):  # Toggle enhancement
                enhance_enabled = not enhance_enabled
                print(f"[INFO] Enhancement: {'ON' if enhance_enabled else 'OFF'}")
            elif key == ord('f') or key == ord('F'):  # Toggle fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                    cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_NORMAL else cv2.WINDOW_NORMAL)
                print("[INFO] Toggled fullscreen mode")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 🧹 CLEANUP
        print("\n[INFO] Shutting down HD system...")
        alarm.stop()
        
        if cap is not None:
            cap.release()
            
        cv2.destroyAllWindows()
        
        if 'face_mesh' in locals():
            face_mesh.close()
            
        print("[INFO] HD system shutdown complete")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)