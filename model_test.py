import cv2
import numpy as np
import mediapipe as mp
import pickle

# Loading model
model = pickle.load(open('model.pkl', 'rb'))

# Initialise mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label class dictionary
class_dict = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 
    15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z',
    27: 'space', 28: 'nothing'
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # x_ = []
    # y_ = []
    # data_aux = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            x_ = []
            y_ = []
            data_aux = []

            # Draw landmarks and connection lines on hands
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Check if image as 21 landmarks
            if len(hand_landmarks.landmark) != 21:
                continue

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

            
        prediction = model.predict([np.asarray(data_aux)])
        
        # Coordinates for bounding boxes
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)
        
        # Create bouding boxes around hands
        cv2.rectangle(frame, (x1-20, y1-20), (x2+20, y2+20), (255, 255, 0), 2)
        cv2.putText(frame, prediction[0], (x1-20, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()