import cv2
import tensorboard
import tensorflow as tf
import keras
from tqdm import tqdm
import numpy as np
from random import shuffle
import time
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = np.load("dataset1.npy", allow_pickle = True)


train, test = train_test_split(data, train_size = 0.9, shuffle = True)

X = np.array([i[0] for i in train]).reshape(-1, 12, IMG_SIZE, IMG_SIZE, 3)
Y = np.array([i[1] for i in train])

x_valid = np.array([i[0] for i in test]).reshape(-1, 12, IMG_SIZE, IMG_SIZE, 3)
y_valid = np.array([i[1] for i in test])

# Normalizing data
X = X.astype('float32')/255
x_valid = x_valid.astype('float32')/255

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(64, (3, 3), activation = "relu", input_shape = (12, IMG_SIZE, IMG_SIZE, 3), padding="same"))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling3D((1,2,2)))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling3D((1,2,2)))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling3D((1,2,2)))

#model.summary()
model.add(Reshape((12, 43264)))

#BiLSTM layers
lstm_fw = LSTM(units = 32)
lstm_bw = LSTM(units = 32, go_backwards = True)
model.add(Bidirectional(lstm_fw, backward_layer = lstm_bw))

# Dense layers
model.add(Dense(64, activation = "relu"))
model.add(Dense(32, activation = "relu"))
model.add(Dense(6, activation = "softmax"))

model.summary()

from tensorflow.keras.optimizers import Adam
INIT_LR = 1e-4
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss=LOSS, optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
NAME = "CNN-BiLSTM-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir = "logs/{}".format(NAME))

model.fit(X, Y, epochs = EPOCHS, validation_data=(x_valid, y_valid),
          batch_size = BATCH_SIZE, verbose=1, callbacks = [tensorboard])


# saving model 
hist = model.save('CNN-BiLSTM.h5', overwrite=True, include_optimizer=True)

# ploting the loss data

plt.plot(his.history["loss"], label='Training loss')
plt.plot(his.history["val_loss"], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig("/content/drive/MyDrive/Face Mask/Training and validation loss.png")


# ploting the accuracy data

plt.plot(his.history["accuracy"], label='Training accuracy')
plt.plot(his.history["val_accuracy"], label='Validation acuuracy')
plt.title('Training and validation accuracy')
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()
plt.savefig("/content/drive/MyDrive/Face Mask/Training and validation ACCURACY.png")

