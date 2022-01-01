import time
import glob
import cv2
import numpy as np

webcam = cv2.VideoCapture(1)
bezel_img = [cv2.imread(file) for file in glob.glob(r"C:\Users\paula\Desktop\Curs de Python\Python\NarutoxBoruto\detectormaska\Poze\*.jpg")]
while True:
    timer = cv2.getTickCount()
    Ok, frame = webcam.read()
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    c = cv2.putText(frame, str(int(float(fps)))+' FPS', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    frame = cv2.flip(frame, 1)
    result = cv2.matchTemplate(frame, bezel_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= 0.4:
        w = bezel_img.shape[1]
        h = bezel_img.shape[0]
        q = cv2.rectangle(frame, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)
        cv2.putText(q, '90160-725/0010 (D5)', (200, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
