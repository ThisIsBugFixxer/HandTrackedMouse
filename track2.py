import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

c = 0

prev_landmarks = None
directions = []

while True:
    success, img = cap.read()

    if not success:
        c += 1
        if c == 1000:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        continue

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
            hand_displacement = sum(
                [abs(x - prev_x) + abs(y - prev_y) for (x, y), (prev_x, prev_y) in zip(landmarks, prev_landmarks)])
            if hand_displacement < 15:
                direction = "Still"
            else:
                movement = (landmarks[0][0] - prev_landmarks[0][0], landmarks[0][1] - prev_landmarks[0][1])
                if abs(movement[0]) > abs(movement[1]):
                    direction = "Horizontal"
                else:
                    direction = "Vertical"
        else:
            direction = "Still"

        if not directions or directions[-1] != direction:
            directions.append(direction)
            if len(directions) > 10:
                directions.pop(0)

            print(f"Direction: {direction}")

        prev_landmarks = landmarks.copy()

        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
