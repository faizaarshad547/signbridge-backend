import cv2
import mediapipe as mp
import csv
import os
import glob

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATASET_PATH = "C:/Users/Bwp Computers/Downloads/archive (2)/asl_alphabet_train/asl_alphabet_train"
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
SAMPLES_PER_LETTER = 300

os.makedirs('data', exist_ok=True)
csv_file = open('data/gesture_data_kaggle.csv', 'w', newline='')
writer = csv.writer(csv_file)

total = 0
for letter in LETTERS:
    folder = os.path.join(DATASET_PATH, letter)
    images = glob.glob(folder + "/*.jpg")[:SAMPLES_PER_LETTER]
    count = 0

    print(f"Processing {letter}... ({len(images)} images found)")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            writer.writerow([letter] + landmarks)
            count += 1

    print(f"  Saved {count} samples for {letter}")
    total += count

csv_file.close()
print(f"\nDone! Total samples saved: {total}")
print("File saved to data/gesture_data_kaggle.csv")