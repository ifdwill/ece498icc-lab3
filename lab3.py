import time
import urllib
import urllib2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import gzip


def get_testset():
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2019/lab2_request_dataset.php'
    values = {'request': 'testdata', 'netid':'zhang258'}
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

test_images = get_testset()

#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
