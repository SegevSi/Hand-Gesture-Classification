
import mediapipe as mp
import cv2
import numpy as np 
import tensorflow as tf
# Initiate holistic model 
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# load model 
pose_clc = tf.keras.models.load_model("hand_pose_clc.keras")
def process_lanmarks(landmarks ) :
    lst = []
    for i in range(len(landmarks)) :
        lst.append(landmarks[i].x)
        lst.append(landmarks[i].y)
        lst.append(landmarks[i].z)
    return np.array([lst])

solutions_dict = {0:"dislike",1:"like",2:"middle finger",3:"gun",4:"ok"}

# describe the type of font to be used.
font = cv2.FONT_HERSHEY_SIMPLEX

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make Detections
        results = holistic.process(image)
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.right_hand_landmarks is not  None  :
            landmarksr =  results.right_hand_landmarks.landmark
            hand_pred = np.argmax(pose_clc.predict(process_lanmarks(landmarksr))[0])
            # Get bounding box for right hand
            x_min = int(min([landmarksr[i].x for i in range(len(landmarksr))]) * frame.shape[1])
            x_max = int(max([landmarksr[i].x for i in range(len(landmarksr))]) * frame.shape[1])
            y_min = int(min([landmarksr[i].y for i in range(len(landmarksr))]) * frame.shape[0])
            y_max = int(max([landmarksr[i].y for i in range(len(landmarksr))]) * frame.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, 
                solutions_dict[hand_pred], 
                (x_max,y_min), 
                font, 1, 
                (0, 147, 175), 
                2, 
                cv2.LINE_4)
        
        if results.left_hand_landmarks is  not  None  :
            landmarksl =  results.left_hand_landmarks.landmark
            hand_pred = np.argmax(pose_clc.predict(process_lanmarks(landmarksl))[0])    
            # Get bounding box for left hand
            x_min = int(min([landmarksl[i].x for i in range(len(landmarksl))]) * frame.shape[1])
            x_max = int(max([landmarksl[i].x for i in range(len(landmarksl))]) * frame.shape[1])
            y_min = int(min([landmarksl[i].y for i in range(len(landmarksl))]) * frame.shape[0])
            y_max = int(max([landmarksl[i].y for i in range(len(landmarksl))]) * frame.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, 
                solutions_dict[hand_pred], 
                (x_max,y_min), 
                font, 1, 
                (0, 147, 175), 
                2, 
                cv2.LINE_4)
                
        # Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # left hand 
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    

        cv2.namedWindow('Raw Webcam Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('Raw Webcam Feed', image)

        # Check for the 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    