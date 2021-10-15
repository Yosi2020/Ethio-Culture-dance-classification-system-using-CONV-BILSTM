import cv2
from tqdm import tqdm
import numpy as np
from random import shuffle
import time

def extract_frames(folder):
  c = 0
  for folders in os.listdir(folder):
    print(folders)
    folder_path = os.path.join(folder, folders)
    for files in tqdm(os.listdir(folder_path)):
      path = os.path.join(folder_path, files)
      cap = cv2.VideoCapture(path)
      success = True

      while success:
        success, image = cap.read()
        if not success:
          break

        cv2.resize(cv2.imwrite(OUTPUT_PATH + str(c) + ".jpg", image), (IMG_SIZE, IMG_SIZE))
        c +=1
    print(c)
    print("Done" + folders)


extract_frames(" ") # enter your path on this 