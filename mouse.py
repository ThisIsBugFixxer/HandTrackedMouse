import mediapipe as mp
import cv2
import time
import pyautogui

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

pTime = 0
cTime = 0

prev_landmarks = None

# Get the screen width and height
screen_width, screen_height = pyautogui.size()

while True:
    success, img = cap.read()

    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        landmarks = []
        for handLms in results.multi_hand_landmarks:
            for lm in handLms.landmark:
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

        if prev_landmarks is not None:
            movement = (landmarks[0][0] - prev_landmarks[0][0], landmarks[0][1] - prev_landmarks[0][1])

            # Adjust the cursor movement sensitivity
            cursor_speed = 3
            cursor_x = int(movement[0] * cursor_speed)
            cursor_y = int(movement[1] * cursor_speed)

            new_x = max(0, min(pyautogui.position()[0] + cursor_x, screen_width))
            new_y = max(0, min(pyautogui.position()[1] + cursor_y, screen_height))

            pyautogui.moveTo(new_x, new_y)

        prev_landmarks = landmarks.copy()

        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
