import os
import cv2

# Initialise path to images
data_dir = r"C:\Users\Rose Maria\OneDrive\Documents\Unified Mentor\ASL detection\dataset"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

no_of_classes = 28
dataset_size = 100

data_index = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 
    15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z',
    27: 'space', 28: 'nothing'
}

cap = cv2.VideoCapture(0)

# Create folders for each class
for class_num in range(no_of_classes):
    folder_path = os.path.join(data_dir, data_index[class_num+1])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f'Collecting data of class {data_index[class_num+1]}')

# Start webcam and detect button press
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Press Q when ready", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0)) 
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) == ord('q'):
            break

    # Start capturing images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(folder_path, f'{counter}.jpg'), frame)

        counter += 1 

cap.release()
cv2.destroyAllWindows()
