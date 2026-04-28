import cv2 as cv
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

# 2. Setup MediaPipe
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

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

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

        # Draw landmarks and connections
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                draw_landmarks(frame, hand)
                
                indexTip = hand[8]
                cursor_x = int(indexTip.x * screen_w)
                cursor_y = int(indexTip.y * screen_h)
                pyautogui.moveTo(cursor_x, cursor_y,duration=0)

        cv.imshow('We Drew Hands', frame)
        if cv.waitKey(1) == ord('q'):
            print("\033[32mExiting.... \033[0m")
            break 

          


cam.release()
cv.destroyAllWindows()