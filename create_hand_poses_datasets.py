import mediapipe as mp
import cv2
import pandas as pd 
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def lanmarks_add_to_list(lst:list , landmarks ) -> None:
    for i in range(len(landmarks)) :
        lst.append(landmarks[i].x)
        lst.append(landmarks[i].y)
        lst.append(landmarks[i].z)

def create_hand_pose_dataset(string : str, num_class : int, num_of_samples :int = 1000) -> None :
        count = 0 
        matrix = []
        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                cap = cv2.VideoCapture(0)

                while cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                                break
                        if count == num_of_samples :
                                break
                        # Recolor Feed
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Make Detections
                        results = holistic.process(image)
                        
                        # Recolor image back to BGR for rendering
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        

                        if  results.left_hand_landmarks is not None :
                                landmarksl =  results.left_hand_landmarks.landmark
                                
                                lst1 = [num_class]
                                
                                lanmarks_add_to_list(lst1,landmarksl)
                                
                                matrix.append(lst1)
                                count+=1
                        if results.right_hand_landmarks is not  None  :
                                landmarksr =  results.right_hand_landmarks.landmark
                                
                                lst1 = [num_class]
                                
                                lanmarks_add_to_list(lst1,landmarksr)
                                
                                matrix.append(lst1)
                                count+=1
                        
                        
                        
                        
                        
                        # describe the type of font
                        # to be used.
                        font = cv2.FONT_HERSHEY_SIMPLEX
                
                        # Use putText() method for
                        # inserting text on video
                        cv2.putText(image, 
                                str(count), 
                                (50, 50), 
                                font, 1, 
                                (0, 255, 255), 
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
        data =pd.DataFrame(data=matrix,columns=["type","x0","y0","z0","x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4","x5","y5","z5","x6","y6","z6","x7","y7","z7","x8","y8","z8","x9","y9","z9","x10","y10","z10","x11","y11","z11","x12","y12","z12","x13","y13","z13","x14","y14","z14","x15","y15","z15","x16","y16","z16","x17","y17","z17","x18","y18","z18","x19","y19","z19","x20","y20","z20"])
        data.to_csv(string+".csv",index=False)

if __name__ == "__main__" :
        create_hand_pose_dataset("dislikes_data",0,2225)
        create_hand_pose_dataset("likes_data",1,2203)
        create_hand_pose_dataset("middle_finger_data",2,2998)
        create_hand_pose_dataset("gun_data",3,2074)
        create_hand_pose_dataset("all_good_data",4,2038)