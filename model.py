import csv
import cv2
import numpy as np

# STEP 1: load the data
lines = []
with open('./driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []
correction = 0.2

for line in lines:
  center_path = line[0]
  left_path = line[1]
  right_path = line[2]

  center_filename = center_path.split('\\')[-1]
  left_filename = left_path.split('\\')[-1]
  right_filename = right_path('\\')[-1]

  current_center_path = './IMG/' + center_filename
  current_left_path = './IMG/' + left_filename
  current_right_path = './IMG' + right_filename

  images.append(cv2.imread(current_center_path))
  measurement = float(line[3])
  measurements.append(measurement)

  image.append(cv2.imread(current_left_path))
  measurements.append(measurement + correction)

  image.append(cv2.imread(current_right_path))
  measurements.append(measurement - correction)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
          nb_epoch=1)

model.save('model.h5')
