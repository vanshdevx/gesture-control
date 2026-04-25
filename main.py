import cv2 as cv
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


        # Draw landmarks on the frame
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                h,w,_=frame.shape
                for landmark in hand:
                    x = int(landmark.x * w) 
                    y = int(landmark.y * h)
                    cv.circle(frame, (x,y), 5, (0,255,0), -1)
        cv.imshow('We Drew Hands', frame)
        if cv.waitKey(1) == ord('q'):
            print("Exiting...")
            break 

          


cam.release()
cv.destroyAllWindows()