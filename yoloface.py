import cv2
import dlib
import face_recognition
import numpy as np 
import torch
import time
import pandas as pd

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s6')

# Load face recognition model
face_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
name = input("Enter your name ")

# Capture a single frame from the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
   print("Error: Could not open the webcam")
   exit()

# Capture a single frame from the webcam
ret, frame = cap.read()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not capture a frame")
    cap.release()
    exit()

# Save the captured frame as an image
cv2.imwrite("captured_image.jpg", frame)

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Load the captured frame for face recognition
captured_frame = cv2.imread("captured_image.jpg")
captured_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)

# Check if any face is detected in the captured image
captured_face_locations = face_recognition.face_locations(captured_rgb)
if captured_face_locations:
    captured_face_encodings = face_recognition.face_encodings(captured_rgb, captured_face_locations)
    known_face_encodings.extend(captured_face_encodings)
    known_face_names.append(name)
else:
    print("Error: No face detected in the captured image")
    exit()

# Initialize some variables
font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)
i = 1
previous_width = 0
prev_frame_time = 0

# Create an empty dictionary to store different DataFrames for each distance_text
distance_df_dict = {}

while True:
    _, frame = cap.read()
    
    if frame is None:
        break

    print("frame", i)
    i += 1
    if i % 1 != 0:
        continue
    
    # time when we finish processing for this frame 
    new_frame_time = time.time() 
  
    # Calculating the fps 
    fps = 1 / (new_frame_time - prev_frame_time) 
    prev_frame_time = new_frame_time 
  
    # converting the fps into an integer 
    fps = int(fps) 
  
    # converting the fps to a string so that we can display it on the frame 
    # by using putText function 
    fps = str(fps) 

    # putting the FPS count on the frame 
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
  
    # YOLOv5 object detection
    results = model(frame)
    for detection in results.pred[0]:
        class_id = detection[-1]
        confidence = detection[4]
        if confidence > 0.3:
            # Object detected is a person
            if class_id == 0:#class ID 0 for person in YOLOv5
                x1, y1, x2, y2 = map(int, detection[:4])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1

                # Face recognition
                face = frame[y1:y2, x1:x2]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb)

                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb, face_locations)

                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            name = "Unknown"

                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:#is confidence score high enough
                                name = known_face_names[best_match_index]

                            # Calculate current width
                            current_width = x2 - x1

                            # Display "going near" or "going far" based on width change
                            if current_width > previous_width:
                                distance_text = "Going near"
                            elif current_width < previous_width:
                                distance_text = "Going far"
                            else:
                                distance_text = "Same place"

                            # Update previous width
                            previous_width = current_width

                            # Check if a DataFrame for the current distance_text exists
                            if distance_text not in distance_df_dict:
                                distance_df_dict[distance_text] = pd.DataFrame(columns=["Frame", "Name", "Distance_Text"])

                            # Append the data to the respective DataFrame
                            distance_df_dict[distance_text] = pd.concat([distance_df_dict[distance_text],
                                                                         pd.DataFrame({"Frame": [i], "Name": [name], "Distance_Text": [distance_text]})],
                                                                         ignore_index=True)

                            if name != "Unknown":
                                # Draw bounding box and display name and distance text
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{name} ({distance_text})", (x1, y1 - 10), font, 1, (0, 255, 0), 1)
                            else:
                                # Draw bounding box and display name and distance text
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f"{name} ({distance_text})", (x1, y1 - 10), font, 1, (0, 0, 255), 1)

    
    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Exit loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save each DataFrame to a separate Excel sheet
with pd.ExcelWriter("distance_info.xlsx") as writer:
    for distance_text, distance_df in distance_df_dict.items():
        distance_df.to_excel(writer, sheet_name=distance_text, index=False)

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
