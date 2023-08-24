import mediapipe as mp
import cv2
import time
import pyautogui

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

pTime = 0
cTime = 0

prev_landmarks = None

# Pinch gesture threshold
pinch_distance_threshold = 50  # Adjust this based on your needs

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
            thumb_tip = landmarks[4]  # Thumb tip landmark
            index_tip = landmarks[8]  # Index finger tip landmark

            # Calculate the Euclidean distance between thumb and index finger tips
            distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

            if distance < pinch_distance_threshold:
                pyautogui.click()  # Perform a click action when pinch is detected
            else:
                # Calculate cursor movement based on hand movement
                cursor_dx = (landmarks[8][0] - prev_landmarks[8][0]) * 3  # Horizontal movement
                cursor_dy = (landmarks[8][1] - prev_landmarks[8][1]) * 3  # Vertical movement
                pyautogui.moveRel(cursor_dx, cursor_dy)

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
