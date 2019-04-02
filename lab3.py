import time
import urllib
import urllib2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import requests
import gzip


def get_testset():
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2019/lab2_request_dataset.php'
    values = {'request': 'testdata', 'netid':'zhang258', 'team':'test'}
    r = requests.post(url, data=values, allow_redirects=True)
    filename = r.url.split("/")[-1]
    testset_id = filename.split(".")[0].split("_")[-1]
    with open(filename, 'wb') as f:
        f.write(r.content)
    return load_dataset(filename), testset_id

def load_dataset(path):
    num_img = 1000
    with gzip.open(path, 'rb') as infile:
        data = np.frombuffer(infile.read(), dtype=np.uint8).reshape(num_img, 784)
    return data

def verify_testset(testset_id, prediction):
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2019/lab2_request_dataset.php'
    values = {'request': 'verify', 'netid':'zhang258', 'testset_id': testset_id, 'prediction': prediction}
    r = requests.post(url, data=values, allow_redirects=True)
    print("got here!")
    print(r.text)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


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
print("training done!")

test_images, testset_id = get_testset()

test_images = np.asarray(test_images)

test_images = test_images.reshape(-1,28,28,1).astype(np.float32)
#print(testset_id)

#print(test_images.shape)

predictions = model.predict(test_images)
#print(predictions)
#print(len(predictions))
a = np.zeros(1000)
for i in range(1000):
    a[i] = np.argmax(predictions[i])

b = ""
for digit in a:
    b += str(digit)

print(b)
print(testset_id)
verify_testset(testset_id, b)
