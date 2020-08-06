import numpy as np
import cv2 # version 4.3.0
import os
import tensorflow

model = tensorflow.keras.models.load_model('models/model-v2.h5')

"""Displaying camera frames in a window"""
clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(1)
cv2.namedWindow('FACEMASK DETECTION')
cv2.setMouseCallback('FACEMASK DETECTION', onMouse)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
labels_dict = {0:'MASK', 1:'NO MASK'}
color_dict = {0:(0,255, 0), 1:(0,0,255)}

print('Showing camera feed. Click window or press any key to stop.')
success, frame = cameraCapture.read()
keycode = cv2.waitKey(1)
# if keycode != -1:
#     keycode &= 0xFF
while success and cv2.waitKey(1) == -1 and not clicked:
    success, frame = cameraCapture.read()

    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image 
    frontal_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    profile_faces = profile_face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(frontal_faces) == 0 and len(profile_faces) > 0:
        faces = profile_faces
        print('Profile Face Detected')
    elif len(profile_faces) == 0 and len(frontal_faces) > 0:
        faces = frontal_faces
        print('Frontal Face Detected')
    else:
        faces = []
        print('No Face Detected')

    if len(faces) == 0:
        x = 100
        y = 100
        h = 300
        w = 300
        resized_image = cv2.resize(frame, (150, 150))
        imgfeatures = resized_image.reshape(1, 150, 150, 3)
        imgfeatures = imgfeatures.astype('float32')
        imgfeatures /= 255
        class_probabilities = model.predict(imgfeatures)
        class_idx = [0 if x < 0.5 else 1 for x in class_probabilities]
        classnames = ['mask', 'no_mask']
        print(f'PREDICTION: {classnames[int(class_idx[0])]}')
        label = int(class_idx[0])

        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.imshow('FACEMASK DETECTION', frame)

    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        frame_gray = gray[y:y+h, x:x+w]
        frame_color = frame[y:y+h, x:x+w]
        resized = cv2.resize(frame_color,(150, 150))
        imgfeatures = resized.reshape(1, 150, 150, 3)
        imgfeatures = imgfeatures.astype('float32')
        imgfeatures /= 255
        result=model.predict(imgfeatures)

        classnames =  ['mask', 'no_mask']
        class_idx = [0 if x > 0.5 else 1 for x in result]
        label = int(class_idx[0])

        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    cv2.imshow('FACEMASK DETECTION', frame)

cv2.destroyWindow('FACEMASK DETECTION')
cameraCapture.release()

