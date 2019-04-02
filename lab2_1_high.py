import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape(60000,28,28,1).astype(np.float32)
train_labels = keras.utils.to_categorical(train_labels, 10).astype(np.float32)

test_images = test_images.reshape(-1,28,28,1).astype(np.float32)
test_labels = keras.utils.to_categorical(test_labels, 10).astype(np.float32)

#create model
model = Sequential()
#add layers
#first layer: 5x5 filter, stride = 1, no padding, output (channels, height, width): (3,24,24) 
model.add(Conv2D(filters=3, kernel_size=5, strides = (1,1), padding='same', input_shape=(28, 28, 1), data_format='channels_last', activation='relu'))
#second layer: stride = 2, output: (3,12,12)
model.add(MaxPooling2D(pool_size=(2,2), strides=2, input_shape = (24, 24, 3), data_format='channels_last'))
#third layer: 3x3 filter, stride = 1, no padding, output (channels, height, width): (3,12,12)
model.add(Conv2D(filters=3, kernel_size=3, strides = 1, padding='valid', input_shape = (12, 12, 3), data_format='channels_last', activation='relu'))
#fourth layer: stride = 2, output: (3,6,6)
model.add(MaxPooling2D(pool_size=(2,2), strides=2, input_shape = (6, 6, 3), data_format='channels_last'))
#flatten for fully-connected layer and add the fully-connected layers
model.add(Flatten())
model.add(Dense(100, input_shape=(108,), activation='relu'))
model.add(Dense(50, input_shape=(100,), activation='relu'))
model.add(Dense(10, input_shape=(50,), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, verbose=2)
print("training finished")

loss_values = history.history['loss']
epochs = range(1, len(loss_values)+1)

loss_acc = model.evaluate(test_images, test_labels, verbose=0)
print("\nTest data loss = %0.4f  accuracy = %0.2f%%" % \
  (loss_acc[0], loss_acc[1]*100) )
# print model.summary

plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()