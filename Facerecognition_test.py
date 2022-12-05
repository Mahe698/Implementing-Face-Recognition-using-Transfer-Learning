import cv2
import os
import pickle
import numpy as np
import pickle
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras_vggface import utils

# dimension of images
image_width = 224
image_height = 224

# load the training labels
face_label_filename = 'face-labels.pickle'
with open(face_label_filename, "rb") as \
    f: class_dictionary = pickle.load(f)

class_list = [value for _, value in class_dictionary.items()]
print(class_list)


# “from keras.utils.layer_utils import get_source_inputs” colab error

# for detecting faces
facecascade = cv2.CascadeClassifier(
    '/content/drive/MyDrive/haarcascade_frontalface_default.xml') #add path of haarcascade_frontalface_default.xml here



test_image_filename = f'  ' #add path of test image here

# load the image
imgtest = cv2.imread(test_image_filename, cv2.IMREAD_COLOR)
image_array = np.array(imgtest, "uint8")

# get the faces detected in the image
faces = facecascade.detectMultiScale(imgtest, 
    scaleFactor=1.1, minNeighbors=5)

# if not exactly 1 face is detected, skip this photo
if len(faces) != 1: 
    print(f'---We need exactly 1 face;photo skipped---')
    print()
    

for (x_, y_, w, h) in faces:
    # draw the face detected
    face_detect = cv2.rectangle(
        imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)
    plt.imshow(face_detect)
    plt.show()

    # resize the detected face to 224x224
    size = (image_width, image_height)
    roi = image_array[y_: y_ + h, x_: x_ + w]
    resized_image = cv2.resize(roi, size)

    # prepare the image for prediction
    x = tf.keras.utils.img_to_array(resized_image)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)

    # making prediction
    predicted_prob = model.predict(x)
    print(predicted_prob)
    print(predicted_prob[0].argmax())
    print("Predicted face: " + class_list[predicted_prob[0].argmax()])
    print("============================\n")
