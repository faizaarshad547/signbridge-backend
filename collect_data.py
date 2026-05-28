import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

os.makedirs('data', exist_ok=True)
csv_file = open('data/gesture_data.csv', 'a', newline='')
writer = csv.writer(csv_file)

print("=== GESTURE DATA COLLECTOR ===")
print("Press A, B, C, D, or E to save that gesture")
print("Press Q to quit")

cap = cv2.VideoCapture(0)
counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
current_gesture = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    landmarks_list = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([lm.x, lm.y, lm.z])

    y = 30
    for letter, count in counts.items():
        cv2.putText(frame, f'{letter}: {count}', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30

    if current_gesture:
        cv2.putText(frame, f'Saving: {current_gesture}', (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Collect Data - Press A/B/C/D/E to save, Q to quit', frame)

    key = cv2.waitKey(1) & 0xFF
    letter = chr(key).upper() if key != 255 else None

    if letter in counts and landmarks_list:
        writer.writerow([letter] + landmarks_list)
        counts[letter] += 1
        current_gesture = letter
    elif key == ord('q'):
        break
    else:
        current_gesture = None

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("Done! Counts:", counts)