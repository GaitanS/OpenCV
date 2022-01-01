#importarea librariilor necesare
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
pozitionare_palme = mp.solutions.hands

captura_webcam = cv2.VideoCapture(1)

with pozitionare_palme.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while captura_webcam.isOpened():
        success, frame = captura_webcam.read()

        # BGR 2 RGB (schimbarea imagini in RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Schimbarea imagini in oglina pentru o vedere mai buna sincrona
        image = cv2.flip(image, 1)

        # Detectia din webcam pentru maini in variabila results
        results = hands.process(image)

        # RGB 2 BGR (schimbarea imagini din RGB in BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Printarea algoritmului de detectie pentru a vedea conexiunea ca este continua
        print(results)

        # Rezultatele redari live conditionat de rezultat
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                # trasarea linilor cu caracteristici, culoare, grosime, diametru
                mp_drawing.draw_landmarks(image, hand, pozitionare_palme.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                                          )

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

captura_webcam.release()
cv2.destroyAllWindows()
