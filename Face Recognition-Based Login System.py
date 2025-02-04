#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install face_recognition opencv-python numpy


# In[ ]:


import face_recognition
import cv2
import os
import numpy as np


# In[ ]:


# Folder where known faces are stored (register users)
known_faces_folder = "C:/Users/Vinoth/OneDrive - Quation Solutions Private Limited/Desktop/known_faces_images"  # Set your folder path here

# Arrays to hold encodings and names
known_face_encodings = []
known_face_names = []

# Register users: Load images from the folder and create encodings
for file_name in os.listdir(known_faces_folder):
    if file_name.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(known_faces_folder, file_name)
        image = face_recognition.load_image_file(image_path)
        
        # Get face encodings
        encodings = face_recognition.face_encodings(image)
        
        if encodings:  # If face is found in the image
            known_face_encodings.append(encodings[0])  # Store only the first encoding
            known_face_names.append(os.path.splitext(file_name)[0])  # Use filename as name (without extension)

# Initialize OpenCV video capture (webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not capture image")
        break
    
    # Convert image to RGB (face_recognition library works with RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find faces in the current frame and their encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Loop through the faces found in the frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Check if there is a match in the known faces database
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]  # Recognized name
        
        # Draw a rectangle around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Display the name below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # If the face matches, grant access
        if name != "Unknown":
            print(f"Welcome, {name}! Access Granted.")
        else:
            print("Unknown face. Access Denied.")

    # Show the video frame with the recognized faces
    cv2.imshow("Face Recognition Login System", frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:


cv2.destroyallwindows()


# In[ ]:


video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




