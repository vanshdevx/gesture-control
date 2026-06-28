import cv2 as cv
import time 
import math
import pyautogui
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks import python as mp_tasks
from collections import deque
import os, subprocess, urllib.request

# 1. Download model if needed
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
     urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

# 2. Setup 
base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode = RunningMode.IMAGE,
    num_hands = 2
)



# Drawing up hte lines 
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), # Thumb
    (0,5),(5,6),(6,7),(7,8), # Index
    (0,9),(9,10),(10,11),(11,12), # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20), # Pinky
    (5,9),(9,13),(13,17) # Palm connections 
]


# Draw landmarks on the frame
def draw_landmarks(frame, hand):
    h,w,_=frame.shape
    for start,end in HAND_CONNECTIONS:
        x1 = int(hand[start].x * w)
        y1 = int(hand[start].y * h)
        x2 = int(hand[end].x * w)
        y2 = int(hand[end].y * h)
        cv.line(frame, (x1,y1), (x2,y2),(255,255,255), 2)

        # Draw dots on top of lines
    for landmark in hand:
        x = int(landmark.x * w) 
        y = int(landmark.y * h)
        cv.circle(frame, (x,y), 5, (0,255,0), -1)


def detect_circle_gesture(points):
    if len(points) < 12:
        return None
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    angles = [math.atan2(y - cy, x - cx) for x, y in points]
    total_delta = 0.0
    for i in range(1, len(angles)):
        delta = angles[i] - angles[i - 1]
        if delta <= -math.pi:
            delta += 2 * math.pi
        elif delta > math.pi:
            delta -= 2 * math.pi
        total_delta += delta
    rotations = abs(total_delta) / (2 * math.pi)
    if rotations < 1.0:
        return None
    direction = 'ccw' if total_delta > 0 else 'cw'
    return direction, rotations


def get_mac_volume():
    try:
        out = subprocess.check_output([
            'osascript', '-e', 'output volume of (get volume settings)'
        ])
        return int(out.decode('utf-8').strip())
    except Exception:
        return None


def set_mac_volume(value):
    value = max(0, min(100, int(value)))
    try:
        subprocess.call(['osascript', '-e', f'set volume output volume {value}'])
    except Exception:
        pass


def adjust_volume(direction, step=5):
    current = get_mac_volume()
    if current is None:
        return
    if direction == 'up':
        set_mac_volume(current + step)
    else:
        set_mac_volume(current - step)


MEDIA_APPS = ('Spotify', 'Music', 'VLC')
media_permission_warned = False
media_state_cache = {'playing': None, 'at': 0.0}


def draw_label(frame, text, x, y, color, scale=0.85, thickness=3):
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv.getTextSize(text, font, scale, thickness)
    px, py = int(x - tw / 2), int(y)
    outline_thickness = thickness + 2
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv.putText(
            frame, text, (px + dx, py + dy), font, scale, (0, 0, 0),
            outline_thickness, cv.LINE_AA,
        )
    cv.putText(frame, text, (px, py), font, scale, color, thickness, cv.LINE_AA)


def is_app_running(app_name):
    try:
        out = subprocess.check_output(
            ['osascript', '-e', f'application "{app_name}" is running'],
            stderr=subprocess.DEVNULL,
        )
        return out.decode('utf-8').strip() == 'true'
    except Exception:
        return False


def get_media_playback_state():
    now = time.time()
    if now - media_state_cache['at'] < 0.4:
        return media_state_cache['playing']

    playing = None
    for app in MEDIA_APPS:
        if not is_app_running(app):
            continue
        try:
            if app == 'VLC':
                script = 'tell application "VLC" to playing'
            else:
                script = f'tell application "{app}" to player state as string'
            out = subprocess.check_output(
                ['osascript', '-e', script],
                stderr=subprocess.DEVNULL,
                timeout=1,
            ).decode('utf-8').strip().lower()
            if app == 'VLC':
                playing = out == 'true'
            else:
                playing = out == 'playing'
            break
        except Exception:
            continue

    media_state_cache['playing'] = playing
    media_state_cache['at'] = now
    return playing


def toggle_playing_media():
    global media_permission_warned

    for app in MEDIA_APPS:
        if not is_app_running(app):
            continue
        try:
            subprocess.run(
                ['osascript', '-e', f'tell application "{app}" to playpause'],
                check=True,
                capture_output=True,
                timeout=1,
            )
            if media_state_cache['playing'] is not None:
                media_state_cache['playing'] = not media_state_cache['playing']
            media_state_cache['at'] = time.time()
            return True
        except Exception:
            continue

    try:
        result = subprocess.run(
            ['osascript', '-e', 'tell application "System Events" to key code 16'],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            if media_state_cache['playing'] is not None:
                media_state_cache['playing'] = not media_state_cache['playing']
            media_state_cache['at'] = time.time()
            return True
        if not media_permission_warned and '1002' in result.stderr:
            media_permission_warned = True
            print(
                '\033[33mMedia pause needs Accessibility permission for system-wide control.\033[0m\n'
                'System Settings → Privacy & Security → Accessibility → enable Terminal or Cursor.\n'
                'Or play media in Spotify, Music, or VLC — those work without it.'
            )
    except Exception:
        pass
    return False


def landmark_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_finger_extended(hand, tip, pip, mcp, palm_width):
    return landmark_distance(hand[tip], hand[mcp]) > palm_width * 0.48


def is_click_pinch(hand, pinch_distance_px, pinch_threshold):
    if pinch_distance_px >= pinch_threshold:
        return False
    palm_width = landmark_distance(hand[5], hand[17])
    if palm_width <= 0:
        return False
    return is_finger_extended(hand, 8, 6, 5, palm_width)


def is_fist(hand):
    palm_width = landmark_distance(hand[5], hand[17])
    if palm_width <= 0:
        return False

    fingers = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]
    if any(is_finger_extended(hand, tip, pip, mcp, palm_width) for tip, pip, mcp in fingers):
        return False

    palm_center_x = sum(hand[i].x for i in (0, 5, 9, 13, 17)) / 5.0
    palm_center_y = sum(hand[i].y for i in (0, 5, 9, 13, 17)) / 5.0
    tip_distances = [
        math.hypot(hand[tip].x - palm_center_x, hand[tip].y - palm_center_y)
        for tip in (8, 12, 16, 20)
    ]
    avg_tip_dist = sum(tip_distances) / len(tip_distances)
    if avg_tip_dist > palm_width * 0.60:
        return False
    if any(dist > palm_width * 0.72 for dist in tip_distances):
        return False

    thumb_dist = math.hypot(hand[4].x - palm_center_x, hand[4].y - palm_center_y)
    if thumb_dist > palm_width * 0.75 and landmark_distance(hand[4], hand[8]) > palm_width * 0.55:
        return False

    finger_spread = (
        landmark_distance(hand[8], hand[12]) +
        landmark_distance(hand[12], hand[16]) +
        landmark_distance(hand[16], hand[20])
    )
    if finger_spread > palm_width * 1.8:
        return False

    return True


# 3. Open camera
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit() 


# Set up pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

# Smoothing variables
prev_x, prev_y = 0, 0
SMOOTHING = 4

#Zone - Mapping 
ZONE_LEFT   = 300
ZONE_RIGHT  = 1620
ZONE_TOP    = 100
ZONE_BOTTOM = 900

# Pinching to click variables
pinch_threshold = 40
last_click = 0 
COOLDOWN = 1.0 

# Volume control via circular index motion
index_history = deque(maxlen=40)
last_volume_action = 0
VOLUME_COOLDOWN = 0.5
VOLUME_STEP_PER_CIRCLE = 8

# Fist play/pause
FIST_COOLDOWN = 1.0
FIST_HOLD_FRAMES = 3
last_fist_action = 0
fist_hold_frames = 0
fist_fired_this_gesture = False

prev_index_y = 0    
# 4. Main loop 
with HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break


        # Flip the frame horizontally 
        frame = cv.flip(frame,1)    


        # Convert to mediapipe image 
        rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        mp_image= mp.Image(image_format=mp.ImageFormat.SRGB, data= rgb)
        result = landmarker.detect(mp_image)

        # Detect hand to draw landmarks and move mouse 
        frame_has_fist = False
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                draw_landmarks(frame, hand)
                h,w,_=frame.shape
                thumb = hand[4]
                indexTip = hand[8]
                middleTip = hand[12]
                middle_x = int(middleTip.x * w)
                middle_y = int(middleTip.y * h)
                thumb_x = int(thumb.x * w)
                thumb_y = int(thumb.y * h)
                indexTip_x = int(indexTip.x * w)
                indexTip_y = int(indexTip.y * h)
                mapped_x = (indexTip_x-ZONE_LEFT)/ (ZONE_RIGHT-ZONE_LEFT) * screen_w
                mapped_y = (indexTip_y-ZONE_TOP) / (ZONE_BOTTOM-ZONE_TOP) * screen_h
                distance = ((thumb_x - indexTip_x) ** 2 + (thumb_y - indexTip_y) ** 2) ** 0.5
                distance_middle = ((thumb_x - middle_x) ** 2 + (thumb_y - middle_y) ** 2) ** 0.5

                # Scroll detection when index and middle fingers are up
                # Use the tip and pip landmarks to confirm the finger is extended.
                index_up = indexTip.y < hand[6].y
                middle_up = middleTip.y < hand[10].y

                index_history.append((indexTip_x, indexTip_y))
                circle_gesture = detect_circle_gesture(index_history)

                if is_fist(hand):
                    frame_has_fist = True
                    index_history.clear()
                    playing = media_state_cache['playing']
                    media_label = "PAUSE" if playing else "PLAY"
                    draw_label(frame, media_label, indexTip_x, indexTip_y - 40, (255, 0, 255))

                # Move the mouse pointer with smoothing 
                cursor_x = max(0, min(screen_w, mapped_x))
                cursor_y = max(0, min(screen_h, mapped_y))
                new_x = prev_x + (cursor_x - prev_x) / SMOOTHING
                new_y = prev_y + (cursor_y - prev_y) / SMOOTHING
                prev_x, prev_y = new_x, new_y
                pyautogui.moveTo(new_x, new_y,duration=0)
                

                # Adding the abilitiy of click by pinching 
                if distance < pinch_threshold:
                    now = time.time()
                    if now - last_click > COOLDOWN:
                        pyautogui.click()
                        last_click = now
                    cv.circle(frame, (indexTip_x, indexTip_y), 15, (0,0,255), -1) 
                elif distance_middle < pinch_threshold:
                    now = time.time()
                    if now - last_click > COOLDOWN:
                        pyautogui.rightClick()
                        last_click = now
                    cv.circle(frame, (middle_x, middle_y), 15, (255,0,0), -1)
                elif circle_gesture and index_up and not middle_up:
                    direction, rotations = circle_gesture
                    now = time.time()
                    if now - last_volume_action > VOLUME_COOLDOWN:
                        amount = max(1, int(round(rotations * VOLUME_STEP_PER_CIRCLE)))
                        if direction == 'cw':
                            adjust_volume('down', amount)
                        else:
                            adjust_volume('up', amount)
                        last_volume_action = now
                        index_history.clear()
                    cv.putText(frame, f"VOL {direction.upper()} {rotations:.1f}", (indexTip_x - 60, indexTip_y - 40),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Scroll when index and middle fingers are up
                elif index_up and middle_up : 
                    delta_y = prev_index_y - indexTip_y
                    if abs(delta_y) > 5:
                        pyautogui.scroll(int(delta_y / 20))
                    cv.putText(frame, "SCROLL", (indexTip_x - 30, indexTip_y - 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                prev_index_y = indexTip_y
        if frame_has_fist:
            fist_hold_frames += 1
            now = time.time()
            if (
                fist_hold_frames >= FIST_HOLD_FRAMES
                and not fist_fired_this_gesture
                and now - last_fist_action > FIST_COOLDOWN
            ):
                toggle_playing_media()
                get_media_playback_state()
                last_fist_action = now
                fist_fired_this_gesture = True
        else:
            fist_hold_frames = 0
            fist_fired_this_gesture = False
        cv.imshow('We Drew Hands', frame)
        if cv.waitKey(1) == ord('q'):
            print("\033[32mExiting.... \033[0m")
            break 

          


cam.release()
cv.destroyAllWindows()
