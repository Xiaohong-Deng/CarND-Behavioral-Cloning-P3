import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# STEP 1-a: load the image paths and steering angles
lines = []
with open('./driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

# STEP 1-b: split the data to training and validation set
# note still haven't load the images to memory
train_samples, valid_samples = train_test_split(lines, test_size=0.2)

correction = 0.2

# STEP 1-c load images batch by batch, save the memory
def generator(samples, batch_size=32):
  num_samples = len(samples)

  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset + batch_size]

      images = []
      measurements = []

      for line in batch_samples:
        measurement = float(line[3])
        left_measurement = measurement + correction
        right_measurement = measurement - correction

        for i in range(3):
          source_path = line[i]
          filename = source_path.split('\\')[-1]
          current_path = './IMG/' + filename
          image_bgr = cv2.imread(current_path)
          image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

          images.append(image)
          images.append(cv2.flip(image, 1))

        measurements.extend([measurement, measurement*-1,
                            left_measurement, left_measurement*-1,
                            right_measurement, right_measurement*-1])

      features = np.array(images)
      labels = np.array(measurements)
      yield shuffle(features, labels)


batch_size = 32

train_generator = generator(train_samples, batch_size=batch_size)
valid_generator = generator(valid_samples, batch_size=batch_size)

# STEP 2: build the network
ch, row, col = 3, 160, 320
nb_batches_train = len(train_samples) // batch_size
nb_batches_valid = len(valid_samples) // batch_size

# NVidia network, 1-node output layer outputing the steering angle
model = Sequential()
model.add(Cropping2D(cropping=((60, 20), (0, 0)),
                     input_shape=(row, col, ch)))
# normalizing the input data
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# default padding='valid' which crops the image if the filter doesnt fit in the last column and row
model.add(Conv2D(24, 5, strides=(2, 2), activation='elu'))
model.add(Dropout(0.8))

model.add(Conv2D(36, 5, strides=(2, 2), activation='elu'))
model.add(Conv2D(48, 5, strides=(2, 2), activation='elu'))
model.add(Conv2D(64, 3, activation='elu'))
model.add(Conv2D(64, 3, activation='elu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# STEP 3: train the network
model.fit_generator(train_generator, steps_per_epoch=nb_batches_train,
                    validation_data=valid_generator,
                    validation_steps=nb_batches_valid,
                    epochs=3)
# STEP 4: save the model
model.save('model.h5')
