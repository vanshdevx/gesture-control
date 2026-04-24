import cv2 as cv
import mediapipe as mp 

cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()  
while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame. Exiting.... *beep* *boop*")
        break
    cv.imshow('Live Feed',frame)
    x = cv.waitKey(1)
    if x == ord('q'):
        print("Exiting.... *beep* *boop*")
        break
cam.release()
cv.destroyAllWindows()
