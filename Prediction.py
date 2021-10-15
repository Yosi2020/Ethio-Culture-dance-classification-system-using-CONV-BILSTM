import tensorflow as tf 
import cv2
import os
import time
import numpy as np
import playsound as ps
from voice import Voice 

os.chdir(r"C:/Users/Eyosiyas/Desktop/bbbbbbb/")

path = r"C:/Users/Eyosiyas/Desktop/my/test/Tigre/61.mp4"
IMG_SIZE = 224
img = []
limit = 0
NUM_FRAMES = 11

CLASSES = ["Afar", "Ahmara", "Gurage", "Oromifa", "Tigre", "Wolaita"]

cap = cv2.VideoCapture(0)
success = True

model = tf.keras.models.load_model(r"C:\Users\Eyosiyas\Desktop\bbbbbbb\CNN-BiLSTM.h5")

while True:
	rect, image = cap.read()
	image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

	img.append(np.array(image))

	if limit == NUM_FRAMES:
		img = np.array(img).reshape(-1, 12, IMG_SIZE, IMG_SIZE,3)
		img = img.astype('float32')/255
		print(img.shape)

		pred = model.predict(img)
		print(pred)
		pre = CLASSES[int(np.argmax(pred[0]))]
		print(pre)
		#Voice(pre)
		img = []
		limit = 0
		NUM_FRAMES = 12

	limit += 1
	cv2.imshow("final", image)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
# release the file pointers
print("[INFO] cleaning up...")
cap.release()
