import os
import re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

# chris evans - 50

# chris_hemsworth 53

# mark_ruffalo 63

# robert_downey_jr 51

# scarlett_johansson 54

os.getcwd()

y_labels = ['chris_evans', 'chris_hemsworth', 'mark_ruffalo', 'robert_downey_jr', 'scarlett_johansson']
X = []
y = []

IMAGE_SIZE = 175


def GetName(imagename):
    return re.sub(r'[0-9]*\.png', '', imagename)


for imagename in os.listdir('../data/images/all'):
    filename = f'../data/images/all/{imagename}'
    hero_name = GetName(imagename)
    image = load_img(filename, target_size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    data = np.array(image)

    # Standardize data

    data = data / 255

    # grayscale the data
    X.append(data[:, :, 0])
    X.append(np.fliplr(data[:, :, 0]))
    y.append(y_labels.index(hero_name))
    y.append(y_labels.index(hero_name))

X = np.array(X)
y = np.array(y)
fig, axs = plt.subplots(5, figsize=(8, 40))

for idx, img in enumerate(X[:5]):
    axs[idx].imshow(img[:, :], cmap='gray')
    axs[idx].set_title(y[idx])

fig

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = X_train.reshape(X_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
X_test = X_test.reshape(X_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# First Crappy Model
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, input_shape=(200, 200, 1), activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32)

model.predict(X_test).argmax(axis=1)

fig, axs = plt.subplots(5, figsize=(8, 40))

for idx, prediction in enumerate(model.predict(X_test).argmax(axis=1)[:5]):
    axs[idx].imshow(X_test[idx], cmap='gray')
    axs[idx].set_title(y_labels[prediction])

fig

# Using CNN

better_model = keras.models.Sequential([
    keras.layers.AveragePooling2D(6, 3, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    keras.layers.Conv2D(64, 3, activation='relu', input_shape=(200, 200, 1)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(5, activation='softmax')
])

better_model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

better_model.fit(X_train, y_train, epochs=5, batch_size=32)

better_model.predict(X_test).argmax(axis=1)

fig, axs = plt.subplots(5, figsize=(8, 40))

for idx, prediction in enumerate(better_model.predict(X_test).argmax(axis=1)[:5]):
    axs[idx].imshow(X_test[idx], cmap='gray')
    axs[idx].set_title(y_labels[prediction])

fig


# Using CNN with Better Layers
better_model = keras.models.Sequential([
    keras.layers.MaxPool2D(2, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    keras.layers.Conv2D(16, kernel_size=(5,5), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
    keras.layers.MaxPool2D((4, 4), padding='same'),
    keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.MaxPool2D((2, 2), padding='same'),
    keras.layers.Conv2D(256, kernel_size=(1,1), activation='relu', padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation='softmax')
])

# keras.layers.MaxPool2D((2, 2), padding='same'),
# keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
# keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
# keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
# keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

better_model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# datagen.fit(X_train)

history = better_model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy",  mode = "max", patience = 10, restore_best_weights = True)], validation_data=(X_test, y_test))

better_model.evaluate(X_test, y_test)
fig, axs = plt.subplots(5, figsize=(8, 40))

for idx, prediction in enumerate(better_model.predict(X_test).argmax(axis=1)[:5]):
    axs[idx].imshow(X_test[idx], cmap='gray')
    axs[idx].set_title(y_labels[prediction])

fig
