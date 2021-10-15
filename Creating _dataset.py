import cv2
from tqdm import tqdm
import numpy as np
from random import shuffle
import time

def create_dataset():
  dataset = []
  images = []
  limit = 0
  count = 0

  for frames in tqdm(os.listdir(r"C:/Users/Eyosiyas/Desktop/my2/")):
    path = os.path.join(r"C:/Users/Eyosiyas/Desktop/my2/", frames)
    img = cv2.resize(cv2.imread(path), (IMG_SIZE, IMG_SIZE))

    images.append(np.array(img))
    limit += 1
    count += 1
    if limit == NUM_FRAMES:
      limit = 0
      if (count < 852):
        dataset.append(np.array([images, np.array([1, 0, 0, 0, 0, 0])]))
      elif ((count >= 852) and (count < 1692)):
        dataset.append(np.array([images, np.array([0, 1, 0, 0, 0, 0])]))
      elif ((count >= 1692) and (count < 2544)):
        dataset.append(np.array([images, np.array([0, 0, 1, 0, 0, 0])]))
      elif((count >= 2544) and (count < 3396)):
        dataset.append(np.array([images, np.array([0, 0, 0, 1, 0, 0])]))
      elif ((count >= 3396) and (count < 4248)) :
        dataset.append(np.array([images, np.array([0, 0, 0, 0, 1, 0])]))
      elif ((count >= 4248) and (count < 5100)):
        dataset.append(np.array([images, np.array([0, 0, 0, 0, 0, 1])]))
      images = []

  shuffle(dataset)
  np.save("dataset1.npy", dataset)
  return dataset 

create_dataset()