import numpy as np
import cv2 # version 4.3.0
import os
import tensorflow

model = tensorflow.keras.models.load_model('models/model-v2.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels_dict = {0:'MASK', 1:'NO MASK'}
color_dict = {0:(0,255, 0), 1:(0,0,255)}

# read an input image
# use you image files
image = cv2.imread('images/input/no_mask.jpg')
image_gray = cv2.imread('images/input/no_mask.jpg', cv2.IMREAD_GRAYSCALE)

# Detects faces of different sizes in the input image 
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

if len(faces) == 0:
    print('No Faces in image')
    resized=cv2.resize(image,(150, 150))
    normalized=resized/255.0
    reshaped=np.reshape(normalized,(1, 150, 150, 3))
    result=model.predict(reshaped)
    class_idx = [0 if x < 0.5 else 1 for x in result]
    classnames =  ['mask', 'no_mask']
    print(f'PREDICTION: {classnames[int(class_idx[0])]}')
else:
    print('Faces. Proceeding...')

for (x,y,w,h) in faces:
    image_color = image[y:y+h, x:x+w]
    resized_image = cv2.resize(image_color, (img_height, img_width))
    imgfeatures = resized_image.reshape(1, 150, 150, 3)
    imgfeatures = imgfeatures.astype('float32')
    imgfeatures /= 255
    class_probabilities = new_model.predict(imgfeatures)
    class_idx = [0 if x < 0.5 else 1 for x in class_probabilities]
    classnames = ['mask', 'no_mask']
    print(f'PREDICTION: {classnames[int(class_idx[0])]}')
    label = int(class_idx[0])

    cv2.rectangle(image,(x,y),(x+w,y+h),color_dict[label],2)
    cv2.rectangle(image,(x,y-40),(x+w,y),color_dict[label],-1)
    cv2.putText(image, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    # creating output images
    cv2.imwrite('output/no_mask.jpg', image)
    image=mpimg.imread('output/no_mask.jpg')
    plt.axis('off')
    plt.imshow(image)

import sys
sys.exit('===========================DONE====================')
