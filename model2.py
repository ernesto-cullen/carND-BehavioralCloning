import csv
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Lambda, Cropping2D
from PIL import Image

samples = []
with open("driving_log.csv", mode="r") as f:
    rows = csv.reader(f)
    next(rows, None)
    for line in rows:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        random.seed(338)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = './' + batch_sample[0]
                name_left = './' + batch_sample[1]
                name_right = './' + batch_sample[2]
                center_image = np.asarray(Image.open(name_center)).reshape(160, 320, 3)
                left_image = np.asarray(Image.open(name_left)).reshape(160, 320, 3)
                right_image = np.asarray(Image.open(name_right)).reshape(160, 320, 3)
                center_angle = float(batch_sample[3])
                left_angle = center_angle+0.3
                right_angle = center_angle-0.3
                rnd = random.random()
                if rnd < 0.33:
                    images.append(left_image)
                    angles.append(left_angle)
                elif rnd < 0.66:
                    images.append(center_image)
                    angles.append(center_angle)
                else:
                    images.append(right_image)
                    angles.append(right_angle)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


BATCH_SIZE = 64
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model = Sequential()
model.add(Cropping2D(cropping=((60, 23), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1.))
model.add(Conv2D(24, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(36, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples) / BATCH_SIZE * 3,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples) / BATCH_SIZE * 3,
                    epochs=5)

model.save('./model.h5')
