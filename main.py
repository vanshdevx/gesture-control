import cv2 as cv
import time 
import pyautogui
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks import python as mp_tasks
import os, urllib.request

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
    num_hands = 1
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


        cv.imshow('We Drew Hands', frame)
        if cv.waitKey(1) == ord('q'):
            print("\033[32mExiting.... \033[0m")
            break 

          


cam.release()
cv.destroyAllWindows()