import os
import cv2
import pickle
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialise mediapipe components
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

dataset_path = r"C:\Projects\Unified Mentor\ASL detection\dataset"

data = []
label = []

for class_path in os.listdir(dataset_path):
    for img_path in os.listdir(os.path.join(dataset_path, class_path)):

        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(dataset_path, class_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # MediaPipe hand detection on image
        results = hands.process(img_rgb)
            
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Collect raw x and y coordinates of each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                # Normalise and store relative position
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

        # Append all landmark coordinates along with their class       
        data.append(data_aux)
        label.append(class_path)

    print(f'Class : {class_path} complete')

# Save data and labels as dictionary in a pickle file
with open('data.pkl', 'wb') as f:
    pickle.dump({'data': data, 'label': label}, f)
    print('Pickle file saved successfully.')