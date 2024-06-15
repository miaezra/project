import keras
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop, SGD, Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import datetime

def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, train_labels = train_images[:300], train_labels[:300]
    test_images, test_labels = test_images[:300], test_labels[:300]
    return train_images, train_labels, test_images, test_labels

def explore_data(train_images, train_labels, test_images, test_labels):
    print("Training set (images) shape: {shape}".format(shape=train_images.shape))
    print("Test set (images) shape: {shape}".format(shape=test_images.shape))
    plt.figure(figsize=[5,5])
    plt.subplot(121)
    plt.imshow(np.reshape(train_images[10], (28,28)), cmap='gray')
    plt.title("(Label: " + str(train_labels[10]) + ")")
    plt.figure(figsize=[5,5])
    plt.subplot(121)
    plt.imshow(np.reshape(train_images[11], (28,28)), cmap='gray')
    plt.title("(Label: " + str(train_labels[11]) + ")")
    plt.show()

def preprocess_data(train_images, test_images):
    train_data = train_images.reshape(-1, 28, 28, 1) / np.max(train_images)
    test_data = test_images.reshape(-1, 28, 28, 1) / np.max(test_images)
    return train_data, test_data

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_data, train_labels, test_data, test_labels):
    train_X, valid_X, train_label, valid_label = train_test_split(train_data, train_labels, test_size=0.2, random_state=13)
    history = model.fit(train_X, train_label, batch_size=64, epochs=100, verbose=1, validation_data=(valid_X, valid_label))
    return history

def evaluate_model(model, history, test_data, test_labels):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    test_eval = model.evaluate(test_data, test_labels, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    predicted_classes = model.predict(test_data)
    predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

    correct = np.where(predicted_classes == test_labels)[0]
    plt.figure(figsize=(10,10))
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3, 3, i+1)
        plt.imshow(test_data[correct].reshape(28, 28), cmap='gray', interpolation='none')
        plt.title(f"Predicted {predicted_classes[correct]}, Class {test_labels[correct]}")
        plt.tight_layout()
    plt.show()

    target_names = [f"Class {i}" for i in range(10)]
    print(classification_report(test_labels, predicted_classes, target_names=target_names))

def main():
    train_images, train_labels, test_images, test_labels = load_data()
    explore_data(train_images, train_labels, test_images, test_labels)
    train_data, test_data = preprocess_data(train_images, test_images)
    model = create_model((28, 28, 1))
    history = train_model(model, train_data, train_labels, test_data, test_labels)
    evaluate_model(model, history, test_data, test_labels)

if __name__ == "__main__":
    main()
