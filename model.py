import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i, line in enumerate(reader):
        # skip the columns titles.
        if (i == 0):
            continue
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

"""
images = []
measurements = []
for i, line in enumerate(lines):
    if (i == 0):
        continue

    # read in the images from center, left, right
    for j in range(3):
        source_path = line[j]
        img_path = './data/' + source_path
        image = cv2.imread(img_path)
        images.append(image)

    # create adjusted steering measurements for the side camera images.
    correction = 0.2 # this is a parameter to adjust.
    steering_center = float(line[3])
    # We add correction to the left because we want it to be closer to center
    # we substract correction from the right because of the same.
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    measurements.extend([steering_center, steering_left, steering_right])

"""

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                # Use images from center, left and right cameras.
                for i in range(3):
                    source_path = batch_sample[i].split('/')[-1]
                    img_path = './data/IMG/' + source_path
                    image = cv2.imread(img_path)
                    images.append(image)

                # create adjusted steering measurements for the side camera images.
                correction = 0.2 # this is a parameter to adjust.
                steering_center = float(batch_sample[3])
                # We add correction to the left because we want it to be closer to center
                # we substract correction from the right because of the same.
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)

            images, angles = augment_data(images, angles)
            x = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(x, y)

def augment_data(images, angles):
    augmented_images, augmented_angles = [], []
    for image, angle in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image, 1))
        augmented_angles.append(angle*-1.0)

    return (augmented_images, augmented_angles)

batch_size = 32
epochs = 5
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Image format.
row, col, ch = 160, 320, 3

# LeNet
model = Sequential()
model.add(
        Lambda(
            lambda x: x / 255.0 - 0.5,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=epochs)
model.save('model.h5')
