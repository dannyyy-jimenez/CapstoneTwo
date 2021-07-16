import os
import re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

formatted_characters = {
    'chris_evans': 'Chris Evans',
    'chris_hemsworth': 'Chris Hemsworth',
    'mark_ruffalo': "Mark Ruffalo",
    'robert_downey_jr': 'Robert Downey JR',
    'scarlett_johansson': "Scarlett Johansson"
}


def PlotConfusionMatrix(model, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    formatted_classes = [formatted_characters[clss] for clss in model.characters]
    plt.imshow(model.cm, interpolation='nearest', cmap=cmap)
    plt.title(f"{model.name} Model Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(formatted_classes))
    plt.xticks(tick_marks, formatted_classes, rotation=45)
    plt.yticks(tick_marks, formatted_classes)

    if normalize:
        cm = model.cm.astype('float') / model.cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm = model.cm
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j].round(2),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'../plots/confusion_matrix_{model.name}.png')


def PlotAccuracy(model):
    """Plot accuracy timelapse of the model based on the epochs

    Parameters
    ----------
    model : Sequential Keras Model

    Returns
    -------
    None

    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(model.history.history['accuracy'], label="train")
    if 'val_accuracy' in model.history.history:
        ax.plot(model.history.history['val_accuracy'], label='test')

    ax.set_title(f'{model.name} Model Accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    fig.legend(loc='upper left')
    fig.savefig(f'../plots/accuracy_{model.name}_model.png')
    pass


def PlotLoss(model):
    """Plot loss timelapse of the model based on the epochs

    Parameters
    ----------
    model : Sequential Keras Model

    Returns
    -------
    None

    """
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.plot(model.history.history['loss'], label='train')
    if 'val_loss' in model.history.history:
        ax.plot(model.history.history['val_loss'], label='test')
        ax.set_ylim([0, 25])

    ax.set_title(f'{model.name} Model Loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    fig.legend(loc='upper left')
    fig.savefig(f'../plots/loss_{model.name}_model.png')


class MCU():
    """Starts our data and prepares it to be put into multiple models

    Parameters
    ----------
    characters : str[]
        Array of characters to look at in the data
    IMAGE_SIZE : int
        Size of images for resizing
    flip : boolean
        flip images to gain more data?
    """

    def __init__(self, characters=['chris_evans', 'chris_hemsworth', 'mark_ruffalo', 'robert_downey_jr', 'scarlett_johansson'], IMAGE_SIZE=175, flip=True):
        self.characters = characters
        self.IMAGE_SIZE = IMAGE_SIZE

        self.X = []
        self.y = []

        imageDist = {}

        for imagename in os.listdir('../data/images/all'):
            filename = f'../data/images/all/{imagename}'
            hero_name = self.GetName(imagename)

            if hero_name not in self.characters:
                continue

            if formatted_characters[hero_name] in imageDist:
                imageDist[formatted_characters[hero_name]] += 1
            else:
                imageDist[formatted_characters[hero_name]] = 1

            image = load_img(filename, target_size=(IMAGE_SIZE, IMAGE_SIZE, 3))
            data = np.array(image)

            # Standardize data
            data = data / 255

            # grayscale the data and flip images lr
            self.X.append(data[:, :, 0])
            self.y.append(self.characters.index(hero_name))
            if flip:
                self.X.append(np.fliplr(data[:, :, 0]))
                self.y.append(self.characters.index(hero_name))

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        self.PlotDist(imageDist)
        self.PlotSampleImages()

    def PlotDist(self, dist):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.set_title("Distribution of Images")
        ax.bar(dist.keys(), dist.values())
        fig.tight_layout()
        fig.savefig(f"../plots/image_dist_{'_'.join(self.characters)}.png")

    def fit(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.y_train = keras.utils.to_categorical(self.y_train)
        self.y_test = keras.utils.to_categorical(self.y_test)

    def PlotSampleImages(self, n=5):
        fig, axs = plt.subplots(n, figsize=(8, n * 8))

        for idx, img in enumerate(self.X[:n]):
            axs[idx].imshow(img[:, :], cmap='gray')
            axs[idx].set_title(formatted_characters[self.characters[self.y[idx]]])

        fig.tight_layout()
        fig.savefig(f'../plots/sample_images_{"-".join(self.characters)}')

    def GetName(self, imagename):
        return re.sub(r'[0-9]*\.png', '', imagename)

# Parent Class for all classifers for code reusability not to be used on its own


class Classifier():
    """Parent class for our classfier models

    Parameters
    ----------
    MCU : MCU
        MCU ready data to process

    Attributes
    ----------
    name : string
        name of our model
    X_test : 2D Array
        2D array of features
    X_train : 2D Array
        2D array of features
    y_train : 1D Array
        Array of labels
    y_test : 1D Array
        Array of labels
    characters : string[]
        the characters used in our model
    IMAGE_SIZE : int
        image sizes
    model : Keras Sequential Model
        dummy model
    history : Keras Model history

    cm : 2D array
        confusion matrix

    """
    def __init__(self, MCU):
        self.name = "Dummy"
        self.X_test = MCU.X_test
        self.X_train = MCU.X_train
        self.y_train = MCU.y_train
        self.y_test = MCU.y_test
        self.characters = MCU.characters
        self.IMAGE_SIZE = MCU.IMAGE_SIZE
        # to be overwritten by child
        self.model = None
        self.history = None
        self.cm = None

    def predictions(self, n=5):
        fig, axs = plt.subplots(n, figsize=(8, n*8))

        for idx, prediction in enumerate(self.model.predict(self.X_test).argmax(axis=1)[:n]):
            axs[idx].imshow(self.X_test[idx], cmap='gray')
            axs[idx].set_title(formatted_characters[self.characters[prediction]])

        fig.tight_layout()
        fig.savefig(f'../plots/{self.name}_model_{"-".join(self.characters)}.png')
        fig


class LogisticRegressionCLF(Classifier):

    def __init__(self, MCU, gridsearch=False):
        super().__init__(MCU)
        self.name = "Logistic"
        self.X_test = self.X_test.reshape(self.X_test.shape[0], MCU.IMAGE_SIZE * MCU.IMAGE_SIZE)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], MCU.IMAGE_SIZE * MCU.IMAGE_SIZE)
        self.y_train = MCU.y_train.argmax(axis=1)
        self.y_test = MCU.y_test.argmax(axis=1)
        self.model = LogisticRegression(penalty='none', tol=0.1, solver='saga', multi_class='multinomial')
        self.history = self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        true_vals = self.y_test
        self.cm = confusion_matrix(y_true=true_vals, y_pred=self.y_pred)

        self.PlotImportantFeatures()
        if gridsearch:
            self.FindBestParams()

    def FindBestParams(self):
        param_grid = {'tol': [0.0001, 0.01, 0.1, 1]}
        clf = GridSearchCV(LogisticRegression(C=1000, penalty='l1', solver='saga', multi_class='multinomial'), param_grid, scoring='accuracy')
        clf.fit(self.X_train, self.y_train)

        print(clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)

        self.model = clf.best_estimator_

    def PlotImportantFeatures(self):
        # The great thing about using Logistic Regression is the interpretability it holds
        # The blue pixel distribution increases prob, whilst red decreases
        scale = np.max(np.abs(self.model.coef_))

        for i in range(len(self.characters)):
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(self.model.coef_[i].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE), cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
            ax.axis('off')
            ax.set_title(f'Pixel Important for {formatted_characters[self.characters[i]]}')
            fig.tight_layout()
            fig.savefig(f'../plots/pixel_imp_{self.characters[i]}')
        pass

    def predictions(self, n=5):
        fig, axs = plt.subplots(n, figsize=(8, n*8))

        for idx, prediction in enumerate(self.model.predict(self.X_test)[:n]):
            axs[idx].imshow(self.X_test[idx].reshape(self.IMAGE_SIZE, self.IMAGE_SIZE), cmap='gray')
            axs[idx].set_title(formatted_characters[self.characters[prediction]])

        fig.tight_layout()
        fig.savefig(f'../plots/logistic_regression_{"-".join(self.characters)}.png')
        fig

    def __str__(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)

        return f"I am a Logistic Regression model with {(accuracy*100).round(2)}% accuracy"


class DefaultNNModel(Classifier):

    def __init__(self, MCU):
        super().__init__(MCU)
        self.name = "Default Neural Network"
        self.model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(512, input_shape=(MCU.IMAGE_SIZE, MCU.IMAGE_SIZE, 1), activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(5, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        self.history = self.model.fit(self.X_train, self.y_train, epochs=45, batch_size=32, validation_data=(self.X_test, self.y_test))
        self.y_pred = self.model.predict(self.X_test).argmax(axis=1)
        true_vals = self.y_test.argmax(axis=1)
        self.cm = confusion_matrix(y_true=true_vals, y_pred=self.y_pred)

    def __str__(self):
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams
        print(self.model.summary())

        accuracy = self.model.evaluate(self.X_test, self.y_test)

        return f"I am a Neural Network model with {totalParams:,} total params\nI performed with {accuracy * 100}% accuracy"


class DefaultCNNModel(Classifier):

    def __init__(self, MCU):
        super().__init__(MCU)
        self.name = "Default Convolutional Neural Network"
        self.model = keras.models.Sequential([
            keras.layers.AveragePooling2D(6, 3, input_shape=(MCU.IMAGE_SIZE, MCU.IMAGE_SIZE, 1)),
            keras.layers.Conv2D(64, 3, activation='relu', input_shape=(MCU.IMAGE_SIZE, MCU.IMAGE_SIZE, 1)),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPool2D(2, 2),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(5, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        self.history = self.model.fit(self.X_train, self.y_train, epochs=45, batch_size=32, validation_data=(self.X_test, self.y_test))
        self.y_pred = self.model.predict(self.X_test).argmax(axis=1)
        true_vals = self.y_test.argmax(axis=1)
        self.cm = confusion_matrix(y_true=true_vals, y_pred=self.y_pred)

    def __str__(self):
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams
        print(self.model.summary())

        accuracy = self.model.evaluate(self.X_test, self.y_test)

        return f"I am a Convolutional Neural Network model with {totalParams:,} total params\nI performed with {accuracy * 100}% accuracy"


class CustomCNNModel(Classifier):

    def __init__(self, MCU):
        super().__init__(MCU)
        self.name = "Custom Convolutional Neural Network"
        self.model = keras.models.Sequential([
            keras.layers.MaxPool2D(2, input_shape=(MCU.IMAGE_SIZE, MCU.IMAGE_SIZE, 1)),
            keras.layers.Conv2D(16, kernel_size=(5,5), activation='relu', input_shape=(MCU.IMAGE_SIZE, MCU.IMAGE_SIZE, 1)),
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

        self.model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        self.history = self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy",  mode = "max", patience = 10, restore_best_weights = True)], validation_data=(self.X_test, self.y_test))
        self.y_pred = self.model.predict(self.X_test).argmax(axis=1)
        true_vals = self.y_test.argmax(axis=1)
        self.cm = confusion_matrix(y_true=true_vals, y_pred=self.y_pred)

    def wrongs(self, n=5):
        wrongs = np.where((self.y_pred == self.y_test) == False)[0]
        to_display = np.random.choice(wrongs, size=n)

        print("Here's some insight as to what the model is getting incorrectly\n")
        print("---------------------------------------------------------------\n\n")

        for wrong in to_display:
            actual = self.y_test[wrong].argmax()
            predicted = self.y_pred[wrong]
            img = self.X_test[wrong]

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f'Predicted Character: {formatted_characters[self.characters[predicted]]}')
            ax.imshow(img, cmap='gray')
            fig.tight_layout()
            fig.savefig(f'../plots/imgclf_wrong_{self.characters[predicted]}')
            print(f'Model Predicted: {formatted_characters[self.characters[predicted]]}')
            print(f'Actual Chracter: {formatted_characters[self.characters[actual]]}')
            print('\n\n\t---------------------------------\n\n')

    def __str__(self):
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams
        print(self.model.summary())

        accuracy = self.model.evaluate(self.X_test, self.y_test)

        return f"I am a Convolutional Neural Network model with {totalParams:,} total params\nI performed with {accuracy * 100}% accuracy"


if __name__ == "__main__":
    mcu = MCU()
    mcu.fit()

    logi = LogisticRegressionCLF(mcu)
